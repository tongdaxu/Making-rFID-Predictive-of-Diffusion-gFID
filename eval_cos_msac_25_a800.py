#!/usr/bin/env python3
"""
Cosine semantic midpoint consistency, adapted to the 25-VAE IFID benchmark.

User-requested definition
-------------------------
For each query latent z_i and its VAE-latent nearest neighbor NN(z_i):

    F1 = DINO(g(z_i))
    F2 = DINO(g(NN(z_i)))
    F3 = DINO(g(SLERP(z_i, NN(z_i), alpha=0.5)))

Then:

    semantic_midpoint_cos
        = E[ cos( (F1 + F2)/2, F3 ) ]

We also report

    semantic_midpoint_cos_dist
        = 1 - semantic_midpoint_cos

so that LOWER is better, like M_SAC / FID, and its correlation with gFID has
the same sign convention.

Important methodological choices
--------------------------------
- NN retrieval: exact whole-latent cosine hard top-1 over all 50k train refs.
- Interpolation: whole-latent SLERP, alpha=0.5.
- F1/F2 use DECODED endpoint latents, exactly as requested:
      g(z_i), g(NN(z_i))
  rather than original raw images.
  This isolates semantic affine consistency of the tokenizer/decoder path from
  ordinary endpoint reconstruction error.
- The selected reference latent is decoded using its OWN cached ImageNet label.
- The interpolated latent is decoded using the query label, consistent with
  the existing IFID path (for decoders that accept labels).
- DINO loader/preprocessing is reused from eval_proxy_dinocos.py.
- Primary semantic readout: x_norm_clstoken.
- Sensitivity check: mean(x_norm_patchtokens).
- Cosine itself performs the final L2 normalization; F1/F2 are averaged before
  that final cosine, exactly matching cos((F1+F2)/2, F3).
- Existing VAE latent cache is STRICTLY READ-ONLY.
- New outputs default to /video_ssd.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset
from ifid.vae.utils import instantiate_from_config

from eval_local_score_rfid_25_fast import (
    encode_vae,
    decode_vae,
    dtype_from_name,
)
from eval_proxy_dinocos import (
    load_dino_encoder,
    preprocess_dinov2_from_m11,
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

    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument(
        "--latent-cache-dir",
        default="/share/xutongda/REPA-E/exps_interpolate/local_score_rfid_25/cache",
        help="Existing raw VAE ref cache; strictly read-only.",
    )
    p.add_argument(
        "--bank-dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
    )
    p.add_argument(
        "--ref-chunk-size",
        type=int,
        default=4096,
        help="Reference chunk size for exact FP32 cosine NN search.",
    )

    p.add_argument("--dino-enc-type", default="dinov2-vit-b")
    p.add_argument(
        "--encoder-code-root",
        default="/share/xutongda/REPA-E",
    )

    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    p.add_argument(
        "--out-dir",
        default=(
            "/video_ssd/kongzishang/xutongda/git/IFID/"
            "cos_msac_results/cos_midpoint_hardcos_slerp_a05_50k50k"
        ),
    )
    return p.parse_args()


def cache_key(args):
    cfg = Path(args.vae_config).stem
    return (
        f"{cfg}_ref{args.ref_size}_res{args.resolution}_seed{args.seed}"
    )


def load_reference_cache_strict(args):
    """
    Load manifest for the existing raw-latent cache. Never writes/rebuilds.
    Returns chunk paths + latent shape.
    """
    root = Path(args.latent_cache_dir) / cache_key(args)
    stats_path = root / "stats.pt"

    if not root.is_dir():
        raise FileNotFoundError(
            f"Missing VAE reference cache directory: {root}"
        )
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing stats.pt: {stats_path}"
        )

    stats = torch.load(stats_path, map_location="cpu")
    if int(stats.get("ref_size", -1)) != args.ref_size:
        raise RuntimeError(
            f"Cache ref_size mismatch: {stats.get('ref_size')} "
            f"vs {args.ref_size}"
        )

    latent_shape = tuple(stats["latent_shape"])
    chunks = sorted(root.glob("chunk*.pt"))
    if not chunks:
        raise RuntimeError(f"No chunk*.pt under {root}")

    # Quick endpoint validation; every chunk is validated again while loading.
    first = torch.load(chunks[0], map_location="cpu")
    last = torch.load(chunks[-1], map_location="cpu")
    if int(first["start"]) != 0:
        raise RuntimeError("Reference cache does not start at 0")
    if int(last["end"]) != args.ref_size:
        raise RuntimeError(
            f"Reference cache ends at {last['end']}, "
            f"expected {args.ref_size}"
        )
    del first, last

    print(
        f"[cache READ-ONLY] {root} "
        f"({len(chunks)} chunks; latent={latent_shape})"
    )
    return chunks, latent_shape, root


@torch.inference_mode()
def load_reference_bank_gpu(
    chunk_paths,
    ref_size,
    latent_shape,
    device,
    dtype,
):
    """
    Load both z and cached ImageNet y. The y bank matters because F2 must be
    g(NN(z_i)), i.e. the selected neighbor decoded under its own label/context.
    """
    z_bank = torch.empty(
        (ref_size, *latent_shape),
        device=device,
        dtype=dtype,
    )
    y_bank = torch.empty(
        (ref_size,),
        device=device,
        dtype=torch.long,
    )

    expected_start = 0

    for p in tqdm(chunk_paths, desc="load raw VAE ref bank -> GPU"):
        pack = torch.load(p, map_location="cpu")

        if "z" not in pack or "y" not in pack:
            raise RuntimeError(
                f"Cache chunk lacks z/y required by this metric: {p}"
            )

        z = pack["z"]
        y = pack["y"].reshape(-1).long()
        start = int(pack["start"])
        end = int(pack["end"])

        if start != expected_start:
            raise RuntimeError(
                f"Non-contiguous cache at {p}: "
                f"start={start}, expected={expected_start}"
            )
        if end - start != z.shape[0] or z.shape[0] != y.shape[0]:
            raise RuntimeError(
                f"Bad z/y sizes in {p}: "
                f"range={end-start}, z={z.shape[0]}, y={y.shape[0]}"
            )
        if tuple(z.shape[1:]) != tuple(latent_shape):
            raise RuntimeError(
                f"Latent shape mismatch in {p}: "
                f"{tuple(z.shape[1:])} vs {latent_shape}"
            )

        z_bank[start:end].copy_(
            z.to(device=device, dtype=dtype, non_blocking=False)
        )
        y_bank[start:end].copy_(
            y.to(device=device, dtype=torch.long, non_blocking=False)
        )

        expected_start = end
        del pack, z, y

    if expected_start != ref_size:
        raise RuntimeError(
            f"Loaded {expected_start} refs, expected {ref_size}"
        )

    return z_bank, y_bank


@torch.inference_mode()
def precompute_ref_norms(ref_bank, chunk):
    N = ref_bank.shape[0]
    norms = torch.empty(
        N, device=ref_bank.device, dtype=torch.float32
    )
    for s in tqdm(
        range(0, N, chunk),
        desc="precompute VAE ref norms",
    ):
        e = min(N, s + chunk)
        rf = ref_bank[s:e].float().reshape(e - s, -1)
        norms[s:e] = rf.norm(dim=1).clamp_min(1e-12)
        del rf
    return norms


@torch.inference_mode()
def exact_global_cosine_nn(
    zq,
    ref_bank,
    ref_norms,
    ref_chunk_size,
):
    B = zq.shape[0]
    q = F.normalize(
        zq.float().reshape(B, -1),
        dim=1,
    )

    best_sim = torch.full(
        (B,), -float("inf"),
        device=zq.device, dtype=torch.float32
    )
    best_idx = torch.zeros(
        (B,), device=zq.device, dtype=torch.long
    )

    N = ref_bank.shape[0]
    for s in range(0, N, ref_chunk_size):
        e = min(N, s + ref_chunk_size)

        rf = ref_bank[s:e].float().reshape(e - s, -1)
        rn = rf / ref_norms[s:e, None]

        sim = q @ rn.T
        vals, pos = sim.max(dim=1)

        improve = vals > best_sim
        best_sim = torch.where(improve, vals, best_sim)
        best_idx = torch.where(
            improve,
            pos.long() + s,
            best_idx,
        )

        del rf, rn, sim, vals, pos, improve

    return best_idx, best_sim


def whole_latent_slerp(z1, z2, t, eps=1e-7):
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
            (1.0 - t) * z1f[small]
            + t * z2f[small]
        )

    return out.reshape_as(z1)


@torch.inference_mode()
def extract_dino_global(encoder, x_m11):
    x = preprocess_dinov2_from_m11(x_m11)
    feats = encoder.forward_features(x)

    if not isinstance(feats, dict):
        raise TypeError(
            f"Expected DINO forward_features dict, got {type(feats)}"
        )
    if "x_norm_clstoken" not in feats:
        raise KeyError(
            "DINO features missing x_norm_clstoken; keys="
            + str(list(feats.keys()))
        )
    if "x_norm_patchtokens" not in feats:
        raise KeyError(
            "DINO features missing x_norm_patchtokens; keys="
            + str(list(feats.keys()))
        )

    cls = feats["x_norm_clstoken"].float()
    meanpatch = feats["x_norm_patchtokens"].float().mean(dim=1)

    return cls, meanpatch


def midpoint_cos(F1, F2, F3, eps=1e-12):
    """
    Exactly:
        cos_sim((F1 + F2)/2, F3)

    No individual endpoint normalization is done before averaging.
    """
    target = 0.5 * (F1 + F2)
    return F.cosine_similarity(
        target,
        F3,
        dim=1,
        eps=eps,
    )


class Running:
    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.ss = 0.0

    def add(self, x):
        x = x.detach().double()
        self.n += int(x.numel())
        self.s += float(x.sum().item())
        self.ss += float((x * x).sum().item())

    def mean(self):
        return self.s / max(self.n, 1)

    def std(self):
        if self.n <= 1:
            return 0.0
        m = self.mean()
        var = max(self.ss / self.n - m * m, 0.0)
        return var ** 0.5


@torch.inference_mode()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1]")
    if args.ref_chunk_size <= 0:
        raise ValueError("--ref-chunk-size must be > 0")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    try:
        torch.set_float32_matmul_precision(
            "high" if args.tf32 else "highest"
        )
    except Exception:
        pass

    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    val_set = ImageNetValDataset(
        args.dataset, transform, args.val_size
    )
    val_loader = DataLoader(
        val_set,
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

    fake_x = torch.zeros(
        1, 3, args.resolution, args.resolution,
        device=device,
    )
    fake_y = torch.zeros(
        1, dtype=torch.long, device=device
    )
    fake_z = encode_vae(vae, fake_x, fake_y).float()
    latent_shape = tuple(fake_z.shape[1:])
    del fake_x, fake_y, fake_z

    chunks, cache_shape, cache_root = (
        load_reference_cache_strict(args)
    )
    if tuple(cache_shape) != latent_shape:
        raise RuntimeError(
            f"VAE/cache latent-shape mismatch: "
            f"{latent_shape} vs {cache_shape}"
        )

    bank_dtype = dtype_from_name(args.bank_dtype)
    ref_bank, ref_y_bank = load_reference_bank_gpu(
        chunks,
        args.ref_size,
        latent_shape,
        device,
        bank_dtype,
    )
    ref_norms = precompute_ref_norms(
        ref_bank,
        args.ref_chunk_size,
    )

    ref_bank_gb = (
        ref_bank.numel() * ref_bank.element_size()
        / (1024.0 ** 3)
    )

    dino, dino_type, dino_arch, loader_module = (
        load_dino_encoder(
            args.dino_enc_type,
            device,
            args.resolution,
            args.encoder_code_root,
        )
    )

    cos_cls = Running()
    cos_meanpatch = Running()
    nn_cos_acc = Running()

    global_seen = 0

    print("=" * 100)
    print("Cosine semantic midpoint consistency")
    print("model              :", args.exp_name)
    print("definition         : cos((F1+F2)/2, F3)")
    print("F1                 : DINO(g(z_i))")
    print("F2                 : DINO(g(NN(z_i)))")
    print("F3                 : DINO(g(SLERP(z_i,NN(z_i),0.5)))")
    print("NN                 : exact whole-latent cosine hard top-1")
    print("ref / val          :", args.ref_size, "/", args.val_size)
    print("latent shape       :", latent_shape)
    print("interpolation      : whole-latent SLERP")
    print("alpha              :", args.alpha)
    print("primary DINO       : x_norm_clstoken")
    print("check DINO         : mean(x_norm_patchtokens)")
    print("endpoint pre-norm  : NONE before midpoint average")
    print("reference cache    :", cache_root, "(READ-ONLY)")
    print("GPU ref bank       :", f"{ref_bank_gb:.2f} GB {ref_bank.dtype}")
    print("DINO               :", dino_type, dino_arch)
    print("DINO loader        :", loader_module)
    print("output             :", out_dir)
    print("=" * 100)

    for img, y, _names in tqdm(
        val_loader,
        desc=f"cos-midpoint {args.exp_name}",
    ):
        B = img.shape[0]

        img = img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        z = encode_vae(vae, img, y).float()

        nn_idx, nn_cos = exact_global_cosine_nn(
            z,
            ref_bank,
            ref_norms,
            args.ref_chunk_size,
        )
        nn_cos_acc.add(nn_cos)

        z_nn = ref_bank.index_select(
            0, nn_idx
        ).float()
        y_nn = ref_y_bank.index_select(
            0, nn_idx
        )

        z_inter = whole_latent_slerp(
            z,
            z_nn,
            args.alpha,
        )

        # F1 = DINO(g(z_i))
        x1 = decode_vae(vae, z, y)
        F1_cls, F1_mean = extract_dino_global(dino, x1)
        del x1

        # F2 = DINO(g(NN(z_i)))
        # Use the selected reference sample's OWN label/context.
        x2 = decode_vae(vae, z_nn, y_nn)
        F2_cls, F2_mean = extract_dino_global(dino, x2)
        del x2

        # F3 = DINO(g(z_interpolate))
        # Keep query label/context, as in the existing IFID path.
        x3 = decode_vae(vae, z_inter, y)
        F3_cls, F3_mean = extract_dino_global(dino, x3)
        del x3

        c_cls = midpoint_cos(
            F1_cls, F2_cls, F3_cls
        )
        c_mean = midpoint_cos(
            F1_mean, F2_mean, F3_mean
        )

        cos_cls.add(c_cls)
        cos_meanpatch.add(c_mean)

        global_seen += B

        del (
            img, y, z, nn_idx, nn_cos, z_nn, y_nn, z_inter,
            F1_cls, F1_mean,
            F2_cls, F2_mean,
            F3_cls, F3_mean,
            c_cls, c_mean,
        )

    if global_seen != args.val_size:
        raise RuntimeError(
            f"Processed {global_seen}, expected {args.val_size}"
        )

    cls_mean = cos_cls.mean()
    meanpatch_mean = cos_meanpatch.mean()

    result = {
        "metric_name": "Cosine Semantic Midpoint Consistency",
        "definition": "E[cos((F1+F2)/2,F3)]",
        "F1": "DINO(g(z_i))",
        "F2": "DINO(g(NN(z_i)))",
        "F3": "DINO(g(SLERP(z_i,NN(z_i),alpha)))",
        "vae_config": args.vae_config,
        "exp_name": args.exp_name,
        "dataset": args.dataset,
        "dataset_ref": args.dataset_ref,
        "ref_size": int(args.ref_size),
        "val_size": int(args.val_size),
        "native_latent_shape": list(latent_shape),
        "nearest_neighbor": {
            "scope": "all reference images",
            "metric": "whole-latent raw cosine",
            "selection": "exact hard top-1",
            "mean_selected_cosine": nn_cos_acc.mean(),
            "std_selected_cosine": nn_cos_acc.std(),
        },
        "interpolation": {
            "type": "whole-latent SLERP",
            "alpha": float(args.alpha),
        },
        "dino": {
            "requested_enc_type": args.dino_enc_type,
            "encoder_type": dino_type,
            "architecture": dino_arch,
            "load_encoders_module": loader_module,
            "primary": "x_norm_clstoken",
            "sensitivity": "mean(x_norm_patchtokens)",
        },
        "metrics": {
            # Higher = better.
            "semantic_midpoint_cos": cls_mean,
            "semantic_midpoint_cos_std": cos_cls.std(),

            # Lower = better; use this when correlating against gFID/FID.
            "semantic_midpoint_cos_dist": 1.0 - cls_mean,

            # Same definition using mean-patch readout.
            "semantic_midpoint_cos_meanpatch": meanpatch_mean,
            "semantic_midpoint_cos_meanpatch_std": cos_meanpatch.std(),
            "semantic_midpoint_cos_dist_meanpatch": 1.0 - meanpatch_mean,
        },
    }

    out_json = out_dir / "metrics.json"
    out_json.write_text(json.dumps(result, indent=2))

    with (out_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key, value in result["metrics"].items():
            w.writerow([key, value])
        w.writerow(
            ["vae_nn_cos_mean", nn_cos_acc.mean()]
        )

    print("\n" + "=" * 100)
    print("RESULT:", args.exp_name)
    print(
        f"semantic midpoint cos (CLS)       = {cls_mean:.8f}"
    )
    print(
        f"semantic midpoint 1-cos (CLS)     = {1.0-cls_mean:.8f}"
    )
    print(
        f"semantic midpoint cos (meanpatch) = {meanpatch_mean:.8f}"
    )
    print(
        f"semantic midpoint 1-cos (meanpatch)= {1.0-meanpatch_mean:.8f}"
    )
    print(
        f"VAE NN cosine mean                = {nn_cos_acc.mean():.8f}"
    )
    print("saved:", out_json)
    print("=" * 100)


if __name__ == "__main__":
    main(parse_args())
