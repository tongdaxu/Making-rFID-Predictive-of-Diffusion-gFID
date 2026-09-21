#!/usr/bin/env python3
"""
Build the shared DINOv2 nearest-neighbor cache for DINO-iFID by REUSING the
repository's already-tested DINO path from eval_proxy_dinocos.py.

Representation
--------------
This intentionally matches the existing `dino_global_cos` definition:

    tokens(x) = forward_features(preprocess(x))["x_norm_patchtokens"]
    f(x)      = L2_normalize(mean(tokens(x), dim=patch))

Then, for every ImageNet-val query image x_i:

    j*(i) = argmax_j f(x_i)^T f(x_j^train)

The output is a single hard top-1 TRAIN IMAGE index per query image.  These
indices are shared by all VAEs.  Interpolation is NOT done here.

Important:
- No Hugging Face model download is used.
- DINO loading and preprocessing are imported directly from
  eval_proxy_dinocos.py:
      load_dino_encoder(...)
      extract_dino_patchtokens(...)
- Therefore the DINO representation is exactly the same repository path used
  by the previous proxy_dinocos experiment.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset

# Reuse the exact existing implementation rather than reimplementing it.
from eval_proxy_dinocos import (
    load_dino_encoder,
    extract_dino_patchtokens,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--dataset-ref", default="/video_ssd/lpm/ImageNet/train")
    p.add_argument("--ref-size", type=int, default=50000)
    p.add_argument("--val-size", type=int, default=50000)
    p.add_argument("--resolution", type=int, default=256)

    # Existing REPA/REPA-E encoder convention.
    p.add_argument("--dino-enc-type", default="dinov2-vit-b")
    p.add_argument(
        "--encoder-code-root",
        default="/share/xutongda/REPA-E",
    )

    # 4090-friendly. Feature extraction is FP32 to match the existing path.
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)

    # Exact 50k-reference cosine search.
    p.add_argument("--nn-query-chunk", type=int, default=256)
    p.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Formal cache default: off, so exact NN matmul is FP32.",
    )

    p.add_argument(
        "--out-dir",
        default=(
            "/share/xutongda/REPA-E/exps_interpolate/"
            "dino_ifid_25/dinov2_vitb_global_ref50000_val50000"
        ),
    )
    p.add_argument(
        "--force-features",
        action="store_true",
        help="Recompute DINO features even if a compatible cache exists.",
    )
    p.add_argument(
        "--force-nn",
        action="store_true",
        help="Recompute exact cosine NN even if cached.",
    )
    return p.parse_args()


def write_names(path, names):
    with Path(path).open("w", encoding="utf-8") as f:
        for name in names:
            f.write(str(name).replace("\n", " ") + "\n")


def read_names(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def cache_is_compatible(meta_path, args):
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        return (
            int(meta["ref_size"]) == int(args.ref_size)
            and int(meta["val_size"]) == int(args.val_size)
            and int(meta["resolution"]) == int(args.resolution)
            and str(meta["dino_model"]) == str(args.dino_enc_type)
            and str(meta["feature"]) == "mean_pool_x_norm_patchtokens"
            and str(meta["nn_metric"]) == "cosine"
            and int(meta["nn_topk"]) == 1
        )
    except Exception:
        return False


@torch.inference_mode()
def extract_global_features(
    loader,
    encoder,
    encoder_type,
    device,
    expected_count,
    desc,
):
    """
    Exact image representation used by existing dino_global_cos:
        normalize(mean(x_norm_patchtokens, dim=1), dim=-1)
    """
    feats_all = []
    names_all = []

    for img, _y, names in tqdm(loader, desc=desc):
        img = img.to(device, non_blocking=True)

        # This function performs the existing repository DINO preprocessing:
        # [-1,1] -> [0,1] -> ImageNet normalize -> bicubic 256->224.
        tokens = extract_dino_patchtokens(
            encoder,
            img,
            encoder_type,
        )

        global_feat = F.normalize(
            tokens.mean(dim=1),
            dim=-1,
        )

        feats_all.append(
            global_feat.cpu().numpy().astype(np.float32, copy=False)
        )
        names_all.extend(str(x) for x in names)

        del img, tokens, global_feat

    feats = np.concatenate(feats_all, axis=0)

    if feats.shape[0] != expected_count:
        raise RuntimeError(
            f"{desc}: got {feats.shape[0]} features, "
            f"expected {expected_count}"
        )
    if len(names_all) != expected_count:
        raise RuntimeError(
            f"{desc}: got {len(names_all)} names, "
            f"expected {expected_count}"
        )

    # Strong sanity: saved features must be unit length.
    norms = np.linalg.norm(feats, axis=1)
    max_norm_err = float(np.max(np.abs(norms - 1.0)))
    if max_norm_err > 2e-5:
        raise RuntimeError(
            f"{desc}: normalized-feature sanity failed, "
            f"max |norm-1|={max_norm_err}"
        )

    return feats, names_all


@torch.inference_mode()
def exact_cosine_top1(ref_np, val_np, device, query_chunk):
    """
    Exact hard top-1 over ALL reference features.
    Both feature arrays are already L2-normalized, so dot == cosine.
    """
    ref = torch.from_numpy(
        np.asarray(ref_np, dtype=np.float32)
    ).to(
        device=device,
        dtype=torch.float32,
    ).contiguous()

    n_query = int(val_np.shape[0])
    best_idx = np.empty(n_query, dtype=np.int64)
    best_sim = np.empty(n_query, dtype=np.float32)

    for qs in tqdm(
        range(0, n_query, query_chunk),
        desc="exact DINO global cosine top-1",
    ):
        qe = min(n_query, qs + query_chunk)

        q_np = np.asarray(val_np[qs:qe], dtype=np.float32)
        q = torch.from_numpy(q_np).to(
            device=device,
            dtype=torch.float32,
        ).contiguous()

        sim = q @ ref.T
        vals, idx = sim.max(dim=1)

        best_idx[qs:qe] = (
            idx.cpu().numpy().astype(np.int64, copy=False)
        )
        best_sim[qs:qe] = (
            vals.cpu().numpy().astype(np.float32, copy=False)
        )

        del q, sim, vals, idx

    return best_idx, best_sim


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.nn_query_chunk <= 0:
        raise ValueError("--nn-query-chunk must be > 0")
    if args.resolution != 256:
        print(
            "[WARN] Existing proxy_dinocos was validated at resolution=256; "
            f"you requested {args.resolution}."
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    try:
        torch.set_float32_matmul_precision(
            "high" if args.tf32 else "highest"
        )
    except Exception:
        pass

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref_feat_path = out / "ref_features.npy"
    val_feat_path = out / "val_features.npy"
    ref_names_path = out / "ref_names.txt"
    val_names_path = out / "val_names.txt"
    nn_idx_path = out / "nn_indices.npy"
    nn_cos_path = out / "nn_cosine.npy"
    meta_path = out / "meta.json"

    compatible = cache_is_compatible(meta_path, args)

    have_features = (
        compatible
        and ref_feat_path.exists()
        and val_feat_path.exists()
        and ref_names_path.exists()
        and val_names_path.exists()
    )

    if args.force_features or not have_features:
        # IMPORTANT: this is exactly the IFID/proxy_dinocos image convention
        # BEFORE extract_dino_patchtokens applies DINO preprocessing.
        transform = transforms.Compose(
            [
                transforms.Resize(args.resolution),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

        ref_set = ImageNetValDataset(
            args.dataset_ref,
            transform,
            args.ref_size,
        )
        val_set = ImageNetValDataset(
            args.dataset,
            transform,
            args.val_size,
        )

        ref_loader = DataLoader(
            ref_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
        )

        dino, dino_type, dino_arch, loader_module = (
            load_dino_encoder(
                args.dino_enc_type,
                device,
                args.resolution,
                args.encoder_code_root,
            )
        )

        print("=" * 88)
        print("Repository DINOv2 NN cache")
        print("requested enc_type  :", args.dino_enc_type)
        print("encoder_type        :", dino_type)
        print("architecture        :", dino_arch)
        print("load_encoders module:", loader_module)
        print("feature             : mean_pool(x_norm_patchtokens)")
        print("feature norm        : L2")
        print("ref / val           :", args.ref_size, "/", args.val_size)
        print("batch size          :", args.batch_size)
        print("resolution          :", args.resolution)
        print("=" * 88)

        ref_feat, ref_names = extract_global_features(
            ref_loader,
            dino,
            dino_type,
            device,
            args.ref_size,
            "DINO ref global features",
        )
        np.save(ref_feat_path, ref_feat)
        write_names(ref_names_path, ref_names)

        val_feat, val_names = extract_global_features(
            val_loader,
            dino,
            dino_type,
            device,
            args.val_size,
            "DINO val global features",
        )
        np.save(val_feat_path, val_feat)
        write_names(val_names_path, val_names)

        meta = {
            "dataset": args.dataset,
            "dataset_ref": args.dataset_ref,
            "ref_size": int(args.ref_size),
            "val_size": int(args.val_size),
            "resolution": int(args.resolution),
            # Keep these names compatible with eval_dino_ifid_25_4090.py.
            "dino_model": args.dino_enc_type,
            "feature": "mean_pool_x_norm_patchtokens",
            "feature_dim": int(ref_feat.shape[1]),
            "feature_normalization": "L2",
            "dino_encoder_type": dino_type,
            "dino_architecture": dino_arch,
            "load_encoders_module": loader_module,
            "encoder_code_root": str(
                Path(args.encoder_code_root).resolve()
            ),
            "source_implementation": (
                "eval_proxy_dinocos.load_dino_encoder + "
                "eval_proxy_dinocos.extract_dino_patchtokens"
            ),
            "preprocessing": (
                "IFID [-1,1] image -> repository DINO preprocessing "
                "([0,1], ImageNet norm, bicubic 256->224)"
            ),
            "nn_metric": "cosine",
            "nn_topk": 1,
            "nn_scope": "all reference images",
            "nn_search_dtype": "float32",
            "tf32_for_nn": bool(args.tf32),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        del dino, ref_feat, val_feat
        torch.cuda.empty_cache()
    else:
        print(f"[cache] reuse compatible DINO features: {out}")

    # If a compatible exact NN cache already exists, reuse it.
    if (
        compatible
        and nn_idx_path.exists()
        and nn_cos_path.exists()
        and not args.force_nn
    ):
        idx = np.load(nn_idx_path)
        cos = np.load(nn_cos_path)

        if (
            idx.shape == (args.val_size,)
            and cos.shape == (args.val_size,)
        ):
            print(f"[cache] reuse DINO NN: {nn_idx_path}")
            print(
                f"NN cosine mean={cos.mean():.6f} "
                f"median={np.median(cos):.6f} "
                f"p05={np.quantile(cos, 0.05):.6f} "
                f"p95={np.quantile(cos, 0.95):.6f}"
            )
            print(
                "unique reference images selected:",
                np.unique(idx).size,
            )
            return

    ref_feat = np.load(ref_feat_path, mmap_mode="r")
    val_feat = np.load(val_feat_path, mmap_mode="r")

    if ref_feat.shape[0] != args.ref_size:
        raise RuntimeError(
            f"ref feature count mismatch: "
            f"{ref_feat.shape[0]} vs {args.ref_size}"
        )
    if val_feat.shape[0] != args.val_size:
        raise RuntimeError(
            f"val feature count mismatch: "
            f"{val_feat.shape[0]} vs {args.val_size}"
        )
    if ref_feat.shape[1] != val_feat.shape[1]:
        raise RuntimeError(
            f"feature dim mismatch: "
            f"{ref_feat.shape} vs {val_feat.shape}"
        )

    idx, cos = exact_cosine_top1(
        ref_feat,
        val_feat,
        device,
        args.nn_query_chunk,
    )

    if idx.min() < 0 or idx.max() >= args.ref_size:
        raise RuntimeError("DINO NN index out of bounds")

    np.save(nn_idx_path, idx)
    np.save(nn_cos_path, cos)

    # Ordering sanity.
    ref_names = read_names(ref_names_path)
    val_names = read_names(val_names_path)
    if len(ref_names) != args.ref_size:
        raise RuntimeError("reference name count mismatch")
    if len(val_names) != args.val_size:
        raise RuntimeError("validation name count mismatch")

    meta = json.loads(meta_path.read_text())
    meta.update(
        {
            "nn_cosine_mean": float(cos.mean()),
            "nn_cosine_median": float(np.median(cos)),
            "nn_cosine_p05": float(np.quantile(cos, 0.05)),
            "nn_cosine_p95": float(np.quantile(cos, 0.95)),
            "unique_reference_images_selected": int(
                np.unique(idx).size
            ),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2))

    print("\n" + "=" * 88)
    print("DINO NN CACHE DONE")
    print("cache             :", out)
    print("feature shape ref :", tuple(ref_feat.shape))
    print("feature shape val :", tuple(val_feat.shape))
    print(
        "NN cosine         : "
        f"mean={cos.mean():.6f}, "
        f"median={np.median(cos):.6f}, "
        f"p05={np.quantile(cos, 0.05):.6f}, "
        f"p95={np.quantile(cos, 0.95):.6f}"
    )
    print(
        "unique selected refs:",
        np.unique(idx).size,
    )
    print("indices           :", nn_idx_path)
    print("=" * 88)


if __name__ == "__main__":
    main(parse_args())
