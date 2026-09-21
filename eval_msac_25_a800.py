#!/usr/bin/env python3
"""
Paper-faithful M_SAC evaluator adapted to the 25-VAE IFID benchmark.

Paper probing:
    X -> z0 = E(X)
    X_nn -> z0_nn = E(X_nn)
    z_alpha = whole-latent SLERP(z0, z0_nn; alpha=0.5)
    Xhat_alpha = D(z_alpha)

    Y      = phi_DINO(X)
    Y_nn   = phi_DINO(X_nn)
    Y_alpha= phi_DINO(Xhat_alpha)

Let
    d = Y_nn - Y
    u = Y_alpha - Y
    alpha_hat = <u,d> / ||d||^2

Then
    delta_parallel = |alpha_hat-alpha| * ||d||_2
    delta_perp     = ||u-alpha_hat*d||_2
    M_SAC          = E[delta_parallel + delta_perp]

This implementation:
- uses exact whole-latent cosine hard top-1 over all reference images;
- uses the original IFID whole-latent SLERP implementation;
- reuses the existing 50k raw VAE reference caches STRICTLY READ-ONLY;
- reuses the repository's existing DINOv2 loader/preprocessing;
- writes results only to /video_ssd by default.

Because the public AffineTok project repository currently contains no released
metric code, two DINO global readouts are reported from one DINO forward:
  * CLS: x_norm_clstoken                    [PRIMARY]
  * mean-patch: mean(x_norm_patchtokens)   [SENSITIVITY CHECK]
Neither readout is post-hoc unit-L2-normalized.
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
        help=(
            "Existing raw VAE reference cache. This evaluator NEVER builds or "
            "modifies it; missing/incomplete cache is a hard error."
        ),
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
        help="Reference chunk used for exact FP32 cosine search.",
    )

    p.add_argument(
        "--semantic-cache",
        default=(
            "/video_ssd/kongzishang/xutongda/git/IFID/"
            "msac_results/dino_semantics_dinov2b_ref50000_val50000"
        ),
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
        help="Default off so hard-NN cosine search is FP32 without TF32.",
    )

    p.add_argument(
        "--out-dir",
        default=(
            "/video_ssd/kongzishang/xutongda/git/IFID/"
            "msac_results/msac_hardcos_slerp_a05_50k50k"
        ),
    )

    return p.parse_args()


def cache_key(args):
    cfg = Path(args.vae_config).stem
    return (
        f"{cfg}_ref{args.ref_size}_res{args.resolution}_seed{args.seed}"
    )


def load_reference_cache_manifest_strict(args):
    """
    Read-only validation of the existing LSM/raw-latent cache.

    Unlike build_or_load_reference_cache(), this function NEVER rebuilds,
    deletes, mkdirs, or writes anything under /share.
    """
    root = Path(args.latent_cache_dir) / cache_key(args)
    stats_path = root / "stats.pt"

    if not root.is_dir():
        raise FileNotFoundError(
            f"Missing VAE reference cache directory: {root}"
        )
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing VAE reference cache stats: {stats_path}"
        )

    stats = torch.load(stats_path, map_location="cpu")
    if int(stats.get("ref_size", -1)) != args.ref_size:
        raise RuntimeError(
            f"Cache ref_size mismatch: "
            f"{stats.get('ref_size')} vs {args.ref_size}"
        )

    latent_shape = tuple(stats["latent_shape"])
    chunk_paths = sorted(root.glob("chunk*.pt"))
    if not chunk_paths:
        raise RuntimeError(f"No cache chunks under {root}")

    # Validate first/last and total contiguity without loading all latent
    # tensors twice. Each actual chunk is validated again during GPU loading.
    first = torch.load(chunk_paths[0], map_location="cpu")
    last = torch.load(chunk_paths[-1], map_location="cpu")

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
        f"({len(chunk_paths)} chunks, latent={latent_shape})"
    )
    return chunk_paths, latent_shape, root


@torch.inference_mode()
def load_raw_reference_bank_gpu(
    chunk_paths,
    ref_size,
    latent_shape,
    device,
    dtype,
):
    bank = torch.empty(
        (ref_size, *latent_shape),
        device=device,
        dtype=dtype,
    )

    expected_start = 0
    for p in tqdm(chunk_paths, desc="load raw VAE ref bank -> GPU"):
        pack = torch.load(p, map_location="cpu")
        z = pack["z"]
        start = int(pack["start"])
        end = int(pack["end"])

        if start != expected_start:
            raise RuntimeError(
                f"Non-contiguous reference cache at {p}: "
                f"start={start}, expected={expected_start}"
            )
        if end - start != z.shape[0]:
            raise RuntimeError(
                f"Bad cache chunk {p}: {end-start} vs {z.shape[0]}"
            )
        if tuple(z.shape[1:]) != tuple(latent_shape):
            raise RuntimeError(
                f"Latent shape mismatch in {p}: "
                f"{tuple(z.shape[1:])} vs {latent_shape}"
            )

        bank[start:end].copy_(
            z.to(device=device, dtype=dtype, non_blocking=False)
        )
        expected_start = end
        del pack, z

    if expected_start != ref_size:
        raise RuntimeError(
            f"Loaded {expected_start} refs, expected {ref_size}"
        )
    return bank


@torch.inference_mode()
def precompute_ref_norms(ref_bank, ref_chunk_size):
    n = ref_bank.shape[0]
    norms = torch.empty(
        n, device=ref_bank.device, dtype=torch.float32
    )
    for s in tqdm(
        range(0, n, ref_chunk_size),
        desc="precompute VAE ref latent norms",
    ):
        e = min(n, s + ref_chunk_size)
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
    """
    Exact hard top-1 cosine NN over ALL reference images.

    Retrieval is whole-latent raw cosine, matching the geometry used for
    SLERP-oriented iFID nearest-neighbor search.
    """
    B = zq.shape[0]
    q = zq.float().reshape(B, -1)
    q = F.normalize(q, dim=1)

    best_sim = torch.full(
        (B,),
        -float("inf"),
        device=zq.device,
        dtype=torch.float32,
    )
    best_idx = torch.zeros(
        (B,),
        device=zq.device,
        dtype=torch.long,
    )

    n = ref_bank.shape[0]
    for s in range(0, n, ref_chunk_size):
        e = min(n, s + ref_chunk_size)
        rf = ref_bank[s:e].float().reshape(e - s, -1)
        rn = rf / ref_norms[s:e, None]
        sim = q @ rn.T
        vals, pos = sim.max(dim=1)

        improve = vals > best_sim
        best_sim = torch.where(improve, vals, best_sim)
        best_idx = torch.where(
            improve,
            pos.to(torch.long) + s,
            best_idx,
        )

        del rf, rn, sim, vals, pos, improve

    return best_idx, best_sim


def whole_latent_slerp(z1, z2, t, eps=1e-7):
    """
    Exact geometry of the original IFID SLERP implementation:
    one angle after flattening the ENTIRE latent.
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

    zt = w1[:, None] * z1f + w2[:, None] * z2f

    small = sin_theta < 1e-6
    if small.any():
        zt[small] = (
            (1.0 - t) * z1f[small] + t * z2f[small]
        )

    return zt.reshape_as(z1)


@torch.inference_mode()
def extract_global_semantics(encoder, x_m11):
    x = preprocess_dinov2_from_m11(x_m11)
    feats = encoder.forward_features(x)

    if not isinstance(feats, dict):
        raise TypeError(
            "Expected DINO forward_features dict, got "
            f"{type(feats)}"
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

    # IMPORTANT for affine Euclidean geometry: no F.normalize here.
    return cls, meanpatch


def load_semantic_cache(args):
    root = Path(args.semantic_cache)
    meta_path = root / "meta.json"

    needed = [
        meta_path,
        root / "ref_cls.npy",
        root / "ref_meanpatch.npy",
        root / "ref_names.txt",
        root / "val_cls.npy",
        root / "val_meanpatch.npy",
        root / "val_names.txt",
    ]
    for p in needed:
        if not p.exists():
            raise FileNotFoundError(p)

    meta = json.loads(meta_path.read_text())
    if int(meta["ref_size"]) != args.ref_size:
        raise RuntimeError("semantic cache ref_size mismatch")
    if int(meta["val_size"]) < args.val_size:
        raise RuntimeError(
            f"semantic cache has only {meta['val_size']} val images; "
            f"need {args.val_size}"
        )
    if int(meta["resolution"]) != args.resolution:
        raise RuntimeError("semantic cache resolution mismatch")
    if meta["dino_enc_type"] != args.dino_enc_type:
        raise RuntimeError(
            f"semantic cache DINO mismatch: "
            f"{meta['dino_enc_type']} vs {args.dino_enc_type}"
        )

    arrays = {
        "ref_cls": np.load(root / "ref_cls.npy", mmap_mode="r"),
        "ref_meanpatch": np.load(
            root / "ref_meanpatch.npy", mmap_mode="r"
        ),
        "val_cls": np.load(root / "val_cls.npy", mmap_mode="r"),
        "val_meanpatch": np.load(
            root / "val_meanpatch.npy", mmap_mode="r"
        ),
    }
    val_names = (root / "val_names.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    ref_names = (root / "ref_names.txt").read_text(
        encoding="utf-8"
    ).splitlines()

    return arrays, val_names, ref_names, meta


def msac_terms(Y, Ynn, Ya, alpha, eps=1e-12):
    """
    Per-sample M_SAC geometry.

    d = Ynn-Y
    u = Ya-Y
    alpha_hat = <u,d>/||d||^2

    Returns:
      delta_parallel, delta_perp, msac, affine_l2, alpha_hat
    """
    d = Ynn - Y
    u = Ya - Y

    d2 = (d * d).sum(dim=1)
    dnorm = torch.sqrt(d2.clamp_min(eps))

    alpha_hat = (u * d).sum(dim=1) / d2.clamp_min(eps)

    delta_parallel = (
        (alpha_hat - float(alpha)).abs() * dnorm
    )

    u_perp = u - alpha_hat[:, None] * d
    delta_perp = u_perp.norm(dim=1)

    msac = delta_parallel + delta_perp

    # The paper also discusses a basic direct-L2 affine deviation.
    # Keep it as a diagnostic, not the primary metric.
    affine_l2 = (
        u - float(alpha) * d
    ).norm(dim=1)

    return (
        delta_parallel,
        delta_perp,
        msac,
        affine_l2,
        alpha_hat,
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
        v = max(self.ss / self.n - m * m, 0.0)
        return v ** 0.5


def metric_accum():
    return {
        "delta_parallel": Running(),
        "delta_perp": Running(),
        "msac": Running(),
        "affine_l2": Running(),
        "alpha_hat": Running(),
    }


def add_terms(acc, terms):
    names = [
        "delta_parallel",
        "delta_perp",
        "msac",
        "affine_l2",
        "alpha_hat",
    ]
    for name, x in zip(names, terms):
        acc[name].add(x)


def summarize_acc(acc):
    out = {}
    for key, stat in acc.items():
        out[key] = stat.mean()
        out[key + "_std"] = stat.std()
    return out


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

    semantics, val_sem_names, ref_sem_names, sem_meta = (
        load_semantic_cache(args)
    )

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

    # Validate semantic-cache ordering against the exact validation dataset.
    for i in (0, args.val_size // 2, args.val_size - 1):
        _x, _y, name = val_set[i]
        if str(name) != val_sem_names[i]:
            raise RuntimeError(
                f"VAL semantic-cache order mismatch at {i}: "
                f"{name!r} vs {val_sem_names[i]!r}"
            )

    # Validate reference semantic-cache names against the ImageNet train
    # enumeration used by the latent cache.
    ref_set = ImageNetValDataset(
        args.dataset_ref, transform, args.ref_size
    )
    for i in (0, args.ref_size // 2, args.ref_size - 1):
        _x, _y, name = ref_set[i]
        if str(name) != ref_sem_names[i]:
            raise RuntimeError(
                f"REF semantic-cache order mismatch at {i}: "
                f"{name!r} vs {ref_sem_names[i]!r}"
            )
    del ref_set

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

    # Probe VAE native latent shape.
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
        load_reference_cache_manifest_strict(args)
    )
    if tuple(cache_shape) != latent_shape:
        raise RuntimeError(
            f"VAE/cache latent-shape mismatch: "
            f"{latent_shape} vs {cache_shape}"
        )

    bank_dtype = dtype_from_name(args.bank_dtype)
    ref_bank = load_raw_reference_bank_gpu(
        chunks,
        args.ref_size,
        latent_shape,
        device,
        bank_dtype,
    )
    ref_bank_gb = (
        ref_bank.numel() * ref_bank.element_size()
        / (1024.0 ** 3)
    )

    ref_norms = precompute_ref_norms(
        ref_bank, args.ref_chunk_size
    )

    # Load the SAME local DINOv2 implementation used in the repository's
    # existing DINO proxy experiments.
    dino, dino_type, dino_arch, loader_module = (
        load_dino_encoder(
            args.dino_enc_type,
            device,
            args.resolution,
            args.encoder_code_root,
        )
    )

    acc_cls = metric_accum()
    acc_mean = metric_accum()
    nn_cos_acc = Running()

    global_pos = 0

    print("=" * 100)
    print("M_SAC evaluator")
    print("model              :", args.exp_name)
    print("latent shape       :", latent_shape)
    print("ref / val          :", args.ref_size, "/", args.val_size)
    print("NN                 : exact whole-latent cosine hard top-1")
    print("interpolation      : whole-latent SLERP")
    print("alpha              :", args.alpha)
    print("semantic primary   : DINO x_norm_clstoken")
    print("semantic sensitivity: mean(x_norm_patchtokens)")
    print("semantic unit norm : NO")
    print("DINO               :", dino_type, dino_arch)
    print("DINO loader        :", loader_module)
    print("reference cache    :", cache_root, "(READ-ONLY)")
    print("GPU ref bank       :", f"{ref_bank_gb:.2f} GB {ref_bank.dtype}")
    print("ref search chunk   :", args.ref_chunk_size)
    print("TF32               :", args.tf32)
    print("output             :", out_dir)
    print("=" * 100)

    for img, y, names in tqdm(
        val_loader,
        desc=f"M_SAC {args.exp_name}",
    ):
        B = img.shape[0]

        # Full ordering verification (cheap relative to the metric).
        for j, name in enumerate(names):
            expected = val_sem_names[global_pos + j]
            if str(name) != expected:
                raise RuntimeError(
                    f"VAL ordering mismatch at {global_pos+j}: "
                    f"{name!r} vs {expected!r}"
                )

        img = img.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)

        z = encode_vae(vae, img, y_dev).float()

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

        z_alpha = whole_latent_slerp(
            z, z_nn, args.alpha
        )

        x_alpha = decode_vae(
            vae, z_alpha, y_dev
        )

        Ya_cls, Ya_mean = extract_global_semantics(
            dino, x_alpha
        )

        idx_np = nn_idx.detach().cpu().numpy().astype(
            np.int64, copy=False
        )

        qslice = slice(global_pos, global_pos + B)

        Y_cls = torch.from_numpy(
            np.asarray(
                semantics["val_cls"][qslice],
                dtype=np.float32,
            )
        ).to(device)
        Ynn_cls = torch.from_numpy(
            np.asarray(
                semantics["ref_cls"][idx_np],
                dtype=np.float32,
            )
        ).to(device)

        Y_mean = torch.from_numpy(
            np.asarray(
                semantics["val_meanpatch"][qslice],
                dtype=np.float32,
            )
        ).to(device)
        Ynn_mean = torch.from_numpy(
            np.asarray(
                semantics["ref_meanpatch"][idx_np],
                dtype=np.float32,
            )
        ).to(device)

        cls_terms = msac_terms(
            Y_cls, Ynn_cls, Ya_cls, args.alpha
        )
        mean_terms = msac_terms(
            Y_mean, Ynn_mean, Ya_mean, args.alpha
        )

        add_terms(acc_cls, cls_terms)
        add_terms(acc_mean, mean_terms)

        global_pos += B

        del (
            img,
            y_dev,
            z,
            nn_idx,
            nn_cos,
            z_nn,
            z_alpha,
            x_alpha,
            Ya_cls,
            Ya_mean,
            Y_cls,
            Ynn_cls,
            Y_mean,
            Ynn_mean,
            cls_terms,
            mean_terms,
        )

    if global_pos != args.val_size:
        raise RuntimeError(
            f"Processed {global_pos}, expected {args.val_size}"
        )

    cls = summarize_acc(acc_cls)
    mean = summarize_acc(acc_mean)

    result = {
        "paper": "AffineTok arXiv:2608.23864",
        "metric_definition": (
            "E[|alpha_hat-alpha|*||d||_2 + "
            "||u-alpha_hat*d||_2]"
        ),
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
            "semantic_cache": str(
                Path(args.semantic_cache).resolve()
            ),
            "semantic_post_unit_normalization": False,
        },
        "metrics": {
            # Primary paper-style global semantic readout.
            "MSAC": cls["msac"],
            "delta_parallel": cls["delta_parallel"],
            "delta_perp": cls["delta_perp"],
            "affine_l2": cls["affine_l2"],
            "alpha_hat_mean": cls["alpha_hat"],
            "MSAC_std": cls["msac_std"],

            # Explicitly named CLS aliases.
            "MSAC_cls": cls["msac"],
            "delta_parallel_cls": cls["delta_parallel"],
            "delta_perp_cls": cls["delta_perp"],
            "affine_l2_cls": cls["affine_l2"],

            # Sensitivity check using the repo's mean-patch global feature.
            "MSAC_meanpatch": mean["msac"],
            "delta_parallel_meanpatch": mean["delta_parallel"],
            "delta_perp_meanpatch": mean["delta_perp"],
            "affine_l2_meanpatch": mean["affine_l2"],
            "alpha_hat_meanpatch_mean": mean["alpha_hat"],
        },
    }

    out_json = out_dir / "metrics.json"
    out_json.write_text(json.dumps(result, indent=2))

    with (out_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in result["metrics"].items():
            w.writerow([k, v])
        w.writerow(
            ["vae_nn_cos_mean", nn_cos_acc.mean()]
        )

    print("\n" + "=" * 100)
    print("RESULT:", args.exp_name)
    print(
        f"MSAC (CLS PRIMARY)      = {cls['msac']:.8f}  "
        f"[parallel={cls['delta_parallel']:.8f}, "
        f"perp={cls['delta_perp']:.8f}]"
    )
    print(
        f"MSAC (mean-patch check)= {mean['msac']:.8f}  "
        f"[parallel={mean['delta_parallel']:.8f}, "
        f"perp={mean['delta_perp']:.8f}]"
    )
    print(
        f"basic affine L2 (CLS)  = {cls['affine_l2']:.8f}"
    )
    print(
        f"VAE NN cosine mean     = {nn_cos_acc.mean():.8f}"
    )
    print("saved:", out_json)
    print("=" * 100)


if __name__ == "__main__":
    main(parse_args())
