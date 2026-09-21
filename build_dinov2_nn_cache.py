#!/usr/bin/env python3
"""
Build a shared DINOv2 nearest-neighbor cache for DINO-iFID.

Definition
----------
For every ImageNet validation query image x_i and every training reference image r_j:

    f_i = normalize(DINOv2(x_i))
    g_j = normalize(DINOv2(r_j))

Use the DINOv2 CLS embedding and exact global cosine hard top-1:

    j*(i) = argmax_j f_i^T g_j

The output NN indices are IMAGE indices only. They are shared by every VAE.
The actual interpolation is NOT done here; it is later done in each VAE's own
encoder latent space.

Default DINO model:
    facebook/dinov2-large  (ViT-L/14, CLS embedding)

Default preprocessing follows the Hugging Face DINOv2 processor:
    Resize shortest edge 256 -> center crop 224 -> ImageNet mean/std.

Outputs
-------
<out-dir>/
    ref_features.npy      normalized float32 DINO features
    val_features.npy      normalized float32 DINO features
    ref_names.txt
    val_names.txt
    nn_indices.npy        int64, shape [val_size]
    nn_cosine.npy         float32, shape [val_size]
    meta.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--dataset-ref", default="/video_ssd/lpm/ImageNet/train")
    p.add_argument("--ref-size", type=int, default=50000)
    p.add_argument("--val-size", type=int, default=50000)

    p.add_argument(
        "--backend",
        choices=["huggingface", "torchhub"],
        default="huggingface",
    )
    p.add_argument(
        "--dino-model",
        default="facebook/dinov2-large",
        help=(
            "Hugging Face model id/path for backend=huggingface; "
            "torch.hub model name for backend=torchhub."
        ),
    )
    p.add_argument(
        "--torchhub-repo",
        default="facebookresearch/dinov2",
        help="GitHub repo or local repo path for backend=torchhub.",
    )
    p.add_argument(
        "--torchhub-source",
        choices=["github", "local"],
        default="github",
    )
    p.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hugging Face: forbid network access and require cached/local weights.",
    )

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--resize-short", type=int, default=256)

    p.add_argument(
        "--amp-dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
    )
    p.add_argument(
        "--nn-query-chunk",
        type=int,
        default=256,
        help="Queries per exact 50k-reference cosine matmul.",
    )
    p.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default off for the formal exact NN search.",
    )

    p.add_argument(
        "--out-dir",
        default=(
            "/share/xutongda/REPA-E/exps_interpolate/"
            "dino_ifid_25/dinov2_large_cls_ref50000_val50000"
        ),
    )
    p.add_argument(
        "--force-features",
        action="store_true",
        help="Recompute DINO features even if valid cache files exist.",
    )
    p.add_argument(
        "--force-nn",
        action="store_true",
        help="Recompute exact cosine NN even if nn_indices.npy exists.",
    )
    return p.parse_args()


def amp_dtype(name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def dino_transform(resize_short=256, image_size=224):
    # Matches facebook/dinov2-large preprocessor defaults.
    return transforms.Compose(
        [
            transforms.Resize(
                resize_short,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_dino(args, device):
    if args.backend == "huggingface":
        try:
            from transformers import AutoModel
        except Exception as e:
            raise RuntimeError(
                "backend=huggingface requires `transformers`. "
                "Install it or use --backend torchhub."
            ) from e

        print(f"[DINO] loading Hugging Face model: {args.dino_model}")
        model = AutoModel.from_pretrained(
            args.dino_model,
            local_files_only=args.local_files_only,
        )
        model = model.to(device).eval()

        @torch.inference_mode()
        def forward_features(x):
            out = model(pixel_values=x)
            if not hasattr(out, "last_hidden_state"):
                raise RuntimeError(
                    "Expected Hugging Face DINOv2 output.last_hidden_state"
                )
            # DINOv2 CLS token.
            return out.last_hidden_state[:, 0]

        return model, forward_features

    print(
        f"[DINO] loading torch.hub model: repo={args.torchhub_repo}, "
        f"model={args.dino_model}, source={args.torchhub_source}"
    )
    model = torch.hub.load(
        args.torchhub_repo,
        args.dino_model,
        source=args.torchhub_source,
        pretrained=True,
    ).to(device).eval()

    @torch.inference_mode()
    def forward_features(x):
        if hasattr(model, "forward_features"):
            out = model.forward_features(x)
            if isinstance(out, dict):
                for key in ("x_norm_clstoken", "x_prenorm"):
                    if key in out:
                        feat = out[key]
                        if key == "x_prenorm" and feat.ndim == 3:
                            feat = feat[:, 0]
                        return feat
            if torch.is_tensor(out):
                if out.ndim == 3:
                    return out[:, 0]
                return out
        out = model(x)
        if torch.is_tensor(out):
            if out.ndim == 3:
                return out[:, 0]
            return out
        raise RuntimeError("Could not extract DINO CLS feature from torch.hub model.")

    return model, forward_features


@torch.inference_mode()
def extract_features(
    loader,
    forward_features,
    device,
    use_amp,
    autocast_dtype,
    expected_count,
    desc,
):
    feats = []
    names_all = []

    for img, _y, names in tqdm(loader, desc=desc):
        img = img.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
            ):
                f = forward_features(img)
        else:
            f = forward_features(img)

        f = F.normalize(f.float(), dim=-1)
        feats.append(f.cpu().numpy().astype(np.float32, copy=False))
        names_all.extend(str(x) for x in names)

    feats = np.concatenate(feats, axis=0)

    if feats.shape[0] != expected_count:
        raise RuntimeError(
            f"{desc}: got {feats.shape[0]} features, expected {expected_count}"
        )
    if len(names_all) != expected_count:
        raise RuntimeError(
            f"{desc}: got {len(names_all)} names, expected {expected_count}"
        )
    return feats, names_all


def write_names(path, names):
    with Path(path).open("w", encoding="utf-8") as f:
        for name in names:
            f.write(str(name).replace("\n", " ") + "\n")


def read_names(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


@torch.inference_mode()
def exact_cosine_top1(ref_np, val_np, device, query_chunk):
    # Features are already L2-normalized float32.
    ref = torch.from_numpy(ref_np).to(
        device=device, dtype=torch.float32
    ).contiguous()

    n_query = val_np.shape[0]
    best_idx = np.empty(n_query, dtype=np.int64)
    best_sim = np.empty(n_query, dtype=np.float32)

    for qs in tqdm(
        range(0, n_query, query_chunk),
        desc="exact DINO cosine top-1",
    ):
        qe = min(n_query, qs + query_chunk)
        q = torch.from_numpy(val_np[qs:qe]).to(
            device=device, dtype=torch.float32
        ).contiguous()

        sim = q @ ref.T
        vals, idx = sim.max(dim=1)

        best_idx[qs:qe] = idx.cpu().numpy().astype(np.int64, copy=False)
        best_sim[qs:qe] = vals.cpu().numpy().astype(np.float32, copy=False)

        del q, sim, vals, idx

    return best_idx, best_sim


def cache_is_compatible(meta_path, args):
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        return (
            int(meta["ref_size"]) == args.ref_size
            and int(meta["val_size"]) == args.val_size
            and meta["backend"] == args.backend
            and meta["dino_model"] == args.dino_model
            and int(meta["resize_short"]) == args.resize_short
            and int(meta["image_size"]) == args.image_size
            and meta["feature"] == "CLS"
        )
    except Exception:
        return False


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    if args.batch_size <= 0 or args.nn_query_chunk <= 0:
        raise ValueError("batch/chunk sizes must be > 0")

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
        transform = dino_transform(
            resize_short=args.resize_short,
            image_size=args.image_size,
        )
        ref_set = ImageNetValDataset(
            args.dataset_ref, transform, args.ref_size
        )
        val_set = ImageNetValDataset(
            args.dataset, transform, args.val_size
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

        model, forward_features = load_dino(args, device)

        adtype = amp_dtype(args.amp_dtype)
        use_amp = adtype != torch.float32

        ref_feat, ref_names = extract_features(
            ref_loader,
            forward_features,
            device,
            use_amp,
            adtype,
            args.ref_size,
            "DINO ref features",
        )
        np.save(ref_feat_path, ref_feat)
        write_names(ref_names_path, ref_names)

        val_feat, val_names = extract_features(
            val_loader,
            forward_features,
            device,
            use_amp,
            adtype,
            args.val_size,
            "DINO val features",
        )
        np.save(val_feat_path, val_feat)
        write_names(val_names_path, val_names)

        del model, forward_features
        torch.cuda.empty_cache()

        meta = {
            "dataset": args.dataset,
            "dataset_ref": args.dataset_ref,
            "ref_size": int(args.ref_size),
            "val_size": int(args.val_size),
            "backend": args.backend,
            "dino_model": args.dino_model,
            "feature": "CLS",
            "feature_dim": int(ref_feat.shape[1]),
            "feature_normalization": "L2",
            "resize_short": int(args.resize_short),
            "image_size": int(args.image_size),
            "preprocess_mean": [0.485, 0.456, 0.406],
            "preprocess_std": [0.229, 0.224, 0.225],
            "nn_metric": "cosine",
            "nn_topk": 1,
            "tf32_for_nn": bool(args.tf32),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
    else:
        print(f"[cache] reuse DINO features: {out}")

    if (
        compatible
        and nn_idx_path.exists()
        and nn_cos_path.exists()
        and not args.force_nn
    ):
        idx = np.load(nn_idx_path)
        cos = np.load(nn_cos_path)
        if idx.shape == (args.val_size,) and cos.shape == (args.val_size,):
            print(f"[cache] reuse DINO NN: {nn_idx_path}")
            print(
                f"NN cosine mean={cos.mean():.6f} "
                f"median={np.median(cos):.6f} "
                f"p05={np.quantile(cos,0.05):.6f} "
                f"p95={np.quantile(cos,0.95):.6f}"
            )
            print(f"unique reference images selected: {np.unique(idx).size}")
            return

    ref_feat = np.load(ref_feat_path, mmap_mode="r")
    val_feat = np.load(val_feat_path, mmap_mode="r")

    if ref_feat.shape[0] != args.ref_size:
        raise RuntimeError(
            f"ref feature count mismatch: {ref_feat.shape[0]} vs {args.ref_size}"
        )
    if val_feat.shape[0] != args.val_size:
        raise RuntimeError(
            f"val feature count mismatch: {val_feat.shape[0]} vs {args.val_size}"
        )
    if ref_feat.shape[1] != val_feat.shape[1]:
        raise RuntimeError(
            f"feature dimension mismatch: {ref_feat.shape} vs {val_feat.shape}"
        )

    idx, cos = exact_cosine_top1(
        ref_feat,
        val_feat,
        device=device,
        query_chunk=args.nn_query_chunk,
    )

    np.save(nn_idx_path, idx)
    np.save(nn_cos_path, cos)

    # Basic index/name sanity.
    ref_names = read_names(ref_names_path)
    val_names = read_names(val_names_path)
    if len(ref_names) != args.ref_size or len(val_names) != args.val_size:
        raise RuntimeError("name cache length mismatch")
    if idx.min() < 0 or idx.max() >= args.ref_size:
        raise RuntimeError("NN index out of bounds")

    # Update meta with NN diagnostics.
    meta = json.loads(meta_path.read_text())
    meta.update(
        {
            "nn_cosine_mean": float(cos.mean()),
            "nn_cosine_median": float(np.median(cos)),
            "nn_cosine_p05": float(np.quantile(cos, 0.05)),
            "nn_cosine_p95": float(np.quantile(cos, 0.95)),
            "unique_reference_images_selected": int(np.unique(idx).size),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2))

    print("\n" + "=" * 80)
    print("DINOv2 NN CACHE DONE")
    print("cache:", out)
    print("ref features:", tuple(ref_feat.shape))
    print("val features:", tuple(val_feat.shape))
    print(
        f"NN cosine: mean={cos.mean():.6f}, "
        f"median={np.median(cos):.6f}, "
        f"p05={np.quantile(cos,0.05):.6f}, "
        f"p95={np.quantile(cos,0.95):.6f}"
    )
    print("unique selected refs:", np.unique(idx).size)
    print("indices:", nn_idx_path)
    print("=" * 80)


if __name__ == "__main__":
    main(parse_args())
