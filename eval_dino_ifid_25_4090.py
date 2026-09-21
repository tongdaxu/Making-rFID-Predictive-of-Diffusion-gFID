#!/usr/bin/env python3
"""
DINO-iFID evaluator.

DINOv2 is used ONLY to select the reference image. Interpolation remains in
each VAE's own encoder latent space.

For query image x_i:
    j*(i) = argmax_j cos(DINO(x_i), DINO(x_j_ref))

Then for the current VAE:
    z_i   = E_VAE(x_i)
    z_j   = E_VAE(x_j*(i))
    z_hat = whole-latent SLERP(z_i, z_j, alpha)
    x_hat = D_VAE(z_hat)

Finally:
    DINO-iFID = FID({x_i}, {x_hat_i})

The same saved DINO NN index j*(i) is reused for every VAE, so all VAEs see
exactly the same image pairs.

This implementation is 4090-friendly:
- no 50k latent bank is resident on GPU;
- query and selected reference images are encoded on-the-fly;
- default batch size is 16.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset
from ifid.fid.fid import InceptionV3, calculate_frechet_distance
from ifid.vae.utils import instantiate_from_config

from eval_local_score_rfid_25_fast import (
    encode_vae,
    decode_vae,
    inception_features,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-config", required=True)
    p.add_argument("--exp-name", required=True)

    p.add_argument("--dataset", default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--dataset-ref", default="/video_ssd/lpm/ImageNet/train")
    p.add_argument("--ref-size", type=int, default=50000)
    p.add_argument("--val-size", type=int, default=50000)
    p.add_argument("--resolution", type=int, default=256)

    p.add_argument(
        "--nn-cache",
        default=(
            "/share/xutongda/REPA-E/exps_interpolate/"
            "dino_ifid_25/dinov2_large_cls_ref50000_val50000"
        ),
    )

    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument(
        "--intp",
        choices=["slerp", "linear"],
        default="slerp",
    )

    p.add_argument(
        "--verify-names",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Verify query/ref dataset ordering against the DINO cache. "
            "Recommended for formal runs."
        ),
    )
    p.add_argument("--save-preview", type=int, default=0)
    p.add_argument("--keep-features", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--out-dir",
        default=(
            "/share/xutongda/REPA-E/exps_interpolate/"
            "dino_ifid_25/dinov2_large_cls_cos_top1_slerp_a05_50k50k"
        ),
    )
    return p.parse_args()


def read_names(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


class DinoPairDataset(Dataset):
    """
    Query image i paired with the training image selected by shared DINO NN.
    """

    def __init__(
        self,
        query_set,
        ref_set,
        nn_indices,
        query_names_cache=None,
        ref_names_cache=None,
        verify_names=True,
    ):
        self.query_set = query_set
        self.ref_set = ref_set
        self.nn_indices = np.asarray(nn_indices, dtype=np.int64)
        self.query_names_cache = query_names_cache
        self.ref_names_cache = ref_names_cache
        self.verify_names = verify_names

        if len(self.query_set) != len(self.nn_indices):
            raise ValueError(
                f"query set / NN index mismatch: "
                f"{len(self.query_set)} vs {len(self.nn_indices)}"
            )

    def __len__(self):
        return len(self.query_set)

    def __getitem__(self, i):
        q_img, q_y, q_name = self.query_set[i]
        r_idx = int(self.nn_indices[i])
        r_img, r_y, r_name = self.ref_set[r_idx]

        if self.verify_names:
            if self.query_names_cache is not None:
                expected_q = self.query_names_cache[i]
                if str(q_name) != expected_q:
                    raise RuntimeError(
                        f"Query order mismatch at {i}: "
                        f"dataset={q_name!r}, DINO-cache={expected_q!r}"
                    )

            if self.ref_names_cache is not None:
                expected_r = self.ref_names_cache[r_idx]
                if str(r_name) != expected_r:
                    raise RuntimeError(
                        f"Reference order mismatch at ref index {r_idx}: "
                        f"dataset={r_name!r}, DINO-cache={expected_r!r}"
                    )

        return (
            q_img,
            q_y,
            str(q_name),
            r_img,
            r_y,
            str(r_name),
            r_idx,
        )


def whole_latent_slerp(
    z1: torch.Tensor,
    z2: torch.Tensor,
    t: float,
    eps: float = 1e-7,
):
    """
    Same whole-latent SLERP geometry used by the recent iFID ablations:
    flatten one complete latent per image, compute one angle, interpolate once,
    then reshape back.
    """
    B = z1.shape[0]
    z1f = z1.float().reshape(B, -1)
    z2f = z2.float().reshape(B, -1)

    z1n = z1f / (z1f.norm(dim=1, keepdim=True) + eps)
    z2n = z2f / (z2f.norm(dim=1, keepdim=True) + eps)

    dot = (z1n * z2n).sum(dim=1).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    w1 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    w2 = torch.sin(t * theta) / (sin_theta + eps)

    out = w1[:, None] * z1f + w2[:, None] * z2f

    small = sin_theta < 1e-6
    if small.any():
        out[small] = (
            (1.0 - t) * z1f[small] + t * z2f[small]
        )

    return out.reshape_as(z1)


def interpolate(zq, zr, mode, alpha):
    if mode == "slerp":
        return whole_latent_slerp(zq, zr, alpha)
    if mode == "linear":
        return (1.0 - alpha) * zq.float() + alpha * zr.float()
    raise ValueError(mode)


def fid_from_features(src, pred):
    mu_src = src.mean(axis=0)
    cov_src = np.cov(src, rowvar=False)
    mu_pred = pred.mean(axis=0)
    cov_pred = np.cov(pred, rowvar=False)
    return float(
        calculate_frechet_distance(
            mu_src, cov_src, mu_pred, cov_pred
        )
    )


def load_nn_cache(args):
    root = Path(args.nn_cache)

    meta_path = root / "meta.json"
    idx_path = root / "nn_indices.npy"
    cos_path = root / "nn_cosine.npy"
    qnames_path = root / "val_names.txt"
    rnames_path = root / "ref_names.txt"

    required = [
        meta_path,
        idx_path,
        cos_path,
        qnames_path,
        rnames_path,
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(
            "DINO NN cache is incomplete:\n" + "\n".join(missing)
        )

    meta = json.loads(meta_path.read_text())
    idx = np.load(idx_path)
    cos = np.load(cos_path)

    if int(meta["ref_size"]) != args.ref_size:
        raise RuntimeError(
            f"DINO cache ref_size={meta['ref_size']} "
            f"but evaluator ref_size={args.ref_size}"
        )
    if int(meta["val_size"]) != args.val_size:
        raise RuntimeError(
            f"DINO cache val_size={meta['val_size']} "
            f"but evaluator val_size={args.val_size}"
        )
    if idx.shape != (args.val_size,):
        raise RuntimeError(
            f"NN index shape mismatch: {idx.shape}"
        )
    if idx.min() < 0 or idx.max() >= args.ref_size:
        raise RuntimeError("DINO NN index out of bounds")

    qnames = read_names(qnames_path)
    rnames = read_names(rnames_path)

    if len(qnames) != args.val_size:
        raise RuntimeError("DINO query-name count mismatch")
    if len(rnames) != args.ref_size:
        raise RuntimeError("DINO ref-name count mismatch")

    return meta, idx, cos, qnames, rnames


@torch.inference_mode()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.bs <= 0:
        raise ValueError("--bs must be > 0")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1]")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)

    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dino_meta, nn_idx, nn_cos, qnames, rnames = load_nn_cache(args)

    vae_transform = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    query_set = ImageNetValDataset(
        args.dataset, vae_transform, args.val_size
    )
    ref_set = ImageNetValDataset(
        args.dataset_ref, vae_transform, args.ref_size
    )

    pair_set = DinoPairDataset(
        query_set=query_set,
        ref_set=ref_set,
        nn_indices=nn_idx,
        query_names_cache=qnames,
        ref_names_cache=rnames,
        verify_names=args.verify_names,
    )

    loader = DataLoader(
        pair_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    vae = instantiate_from_config(
        OmegaConf.load(args.vae_config)
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    src_feats = []
    pred_feats = []

    # Useful diagnostics: how close the DINO-selected pair is inside THIS VAE.
    vae_pair_cos_sum = 0.0
    vae_pair_cos_count = 0
    vae_pair_rel_l2_sum = 0.0

    preview_left = int(args.save_preview)
    preview_dir = out_dir / "preview"
    if preview_left > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("DINO-iFID")
    print("VAE              :", args.exp_name)
    print("DINO model       :", dino_meta["dino_model"])
    print("DINO feature     :", dino_meta["feature"])
    print("DINO NN          : global cosine hard top-1")
    print("same pairs / VAE : YES")
    print("ref / val        :", args.ref_size, "/", args.val_size)
    print("VAE batch size   :", args.bs)
    print(
        "interpolation    : whole-latent",
        args.intp,
        "alpha=",
        args.alpha,
    )
    print(
        "DINO NN cosine   : "
        f"mean={float(nn_cos.mean()):.6f}, "
        f"median={float(np.median(nn_cos)):.6f}"
    )
    print("=" * 88)

    for batch in tqdm(loader, desc=f"DINO-iFID {args.exp_name}"):
        (
            q_img,
            q_y,
            q_name,
            r_img,
            r_y,
            r_name,
            r_idx,
        ) = batch

        q_img = q_img.to(device, non_blocking=True)
        q_y = q_y.to(device, non_blocking=True)
        r_img = r_img.to(device, non_blocking=True)
        r_y = r_y.to(device, non_blocking=True)

        # Encode separately: keeps peak VRAM much lower on a 24-GB 4090.
        zq = encode_vae(vae, q_img, q_y).float()
        zr = encode_vae(vae, r_img, r_y).float()

        if zq.shape != zr.shape:
            raise RuntimeError(
                f"Query/ref latent shape mismatch: {zq.shape} vs {zr.shape}"
            )

        qf = zq.reshape(zq.shape[0], -1)
        rf = zr.reshape(zr.shape[0], -1)

        pair_cos = torch.nn.functional.cosine_similarity(
            qf, rf, dim=1
        )
        pair_rel = (
            torch.linalg.vector_norm(qf - rf, dim=1)
            / torch.linalg.vector_norm(qf, dim=1).clamp_min(1e-12)
        )
        vae_pair_cos_sum += float(pair_cos.sum().item())
        vae_pair_rel_l2_sum += float(pair_rel.sum().item())
        vae_pair_cos_count += int(zq.shape[0])

        zhat = interpolate(
            zq,
            zr,
            mode=args.intp,
            alpha=args.alpha,
        )

        # Match prior iFID code: decode using the QUERY label/context.
        xhat = decode_vae(vae, zhat, q_y)

        q01 = (q_img.float() + 1.0) / 2.0
        h01 = (xhat.float() + 1.0) / 2.0

        src_feats.append(
            inception_features(inception, q01)
            .float()
            .cpu()
            .numpy()
        )
        pred_feats.append(
            inception_features(inception, h01)
            .float()
            .cpu()
            .numpy()
        )

        if preview_left > 0:
            from PIL import Image

            q_cpu = q01.detach().cpu()
            r_cpu = ((r_img.float() + 1.0) / 2.0).detach().cpu()
            h_cpu = h01.detach().cpu()
            nsave = min(preview_left, q_cpu.shape[0])

            def save_tensor(t, path):
                arr = (
                    torch.clamp(t, 0, 1)
                    .mul(255)
                    .round()
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .numpy()
                )
                Image.fromarray(arr).save(path)

            for j in range(nsave):
                stem = Path(str(q_name[j])).stem
                save_tensor(
                    q_cpu[j], preview_dir / f"{stem}_query.png"
                )
                save_tensor(
                    r_cpu[j], preview_dir / f"{stem}_dino_nn.png"
                )
                save_tensor(
                    h_cpu[j], preview_dir / f"{stem}_interp.png"
                )
            preview_left -= nsave

        del (
            q_img,
            q_y,
            r_img,
            r_y,
            zq,
            zr,
            qf,
            rf,
            pair_cos,
            pair_rel,
            zhat,
            xhat,
            q01,
            h01,
        )

    src = np.concatenate(src_feats, axis=0)
    pred = np.concatenate(pred_feats, axis=0)

    if src.shape[0] != args.val_size:
        raise RuntimeError(
            f"Expected {args.val_size} source features, got {src.shape[0]}"
        )
    if pred.shape[0] != args.val_size:
        raise RuntimeError(
            f"Expected {args.val_size} pred features, got {pred.shape[0]}"
        )

    print("[FID] computing covariance/sqrtm ...", flush=True)
    score = fid_from_features(src, pred)

    model_tag = (
        str(dino_meta["dino_model"])
        .split("/")[-1]
        .replace("-", "_")
    )
    metric_key = (
        f"dino_ifid_{model_tag}_cls_cos_top1_"
        f"{args.intp}_a{args.alpha:g}"
    )

    diagnostics = {
        "mean_dino_nn_cosine": float(nn_cos.mean()),
        "median_dino_nn_cosine": float(np.median(nn_cos)),
        "mean_vae_latent_cosine_of_dino_pairs": (
            vae_pair_cos_sum / max(vae_pair_cos_count, 1)
        ),
        "mean_vae_latent_relative_l2_of_dino_pairs": (
            vae_pair_rel_l2_sum / max(vae_pair_cos_count, 1)
        ),
    }

    result = {
        "vae_config": args.vae_config,
        "exp_name": args.exp_name,
        "dataset": args.dataset,
        "dataset_ref": args.dataset_ref,
        "ref_size": int(args.ref_size),
        "val_size": int(args.val_size),
        "dino_nn_cache": str(Path(args.nn_cache).resolve()),
        "dino_model": dino_meta["dino_model"],
        "dino_feature": dino_meta["feature"],
        "dino_nn_metric": "cosine",
        "dino_nn_topk": 1,
        "reference_pairing_shared_across_vaes": True,
        "interpolation_space": "current VAE encoder latent",
        "interpolation": f"whole_latent_{args.intp}",
        "alpha": float(args.alpha),
        "decode_condition": "query label/context",
        "batch_size": int(args.bs),
        "metrics": {metric_key: score},
        "diagnostics": diagnostics,
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2)
    )

    with (out_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow([metric_key, f"{score:.10f}"])
        for k, v in diagnostics.items():
            w.writerow([k, f"{v:.10f}"])

    if args.keep_features:
        np.save(out_dir / "features_source.npy", src)
        np.save(out_dir / "features_dino_ifid.npy", pred)

    print("\n" + "=" * 88)
    print("DONE:", args.exp_name)
    print(f"{metric_key} = {score:.6f}")
    print(
        "VAE latent cosine of DINO pairs = "
        f"{diagnostics['mean_vae_latent_cosine_of_dino_pairs']:.6f}"
    )
    print(
        "VAE latent relative-L2 of DINO pairs = "
        f"{diagnostics['mean_vae_latent_relative_l2_of_dino_pairs']:.6f}"
    )
    print("metrics:", out_dir / "metrics.json")
    print("=" * 88)


if __name__ == "__main__":
    main(parse_args())
