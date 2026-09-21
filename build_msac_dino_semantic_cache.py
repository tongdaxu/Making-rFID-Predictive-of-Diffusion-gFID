#!/usr/bin/env python3
"""
Build a shared DINOv2 semantic cache for M_SAC.

This reuses the exact repository DINO loader/preprocessing already used by
eval_proxy_dinocos.py.  It intentionally stores NON-unit-normalized global
semantic vectors because M_SAC is an Euclidean affine-geometry metric.

For every ImageNet image we cache both:
  1) cls       = forward_features(...)[ "x_norm_clstoken" ]
  2) meanpatch = mean(forward_features(...)[ "x_norm_patchtokens" ], dim=patch)

The paper denotes the semantic extractor simply as phi_DINO.  The current
public project repository does not expose the official M_SAC implementation,
so keeping both global readouts lets us test the paper-natural CLS readout
(primary) and the mean-patch global readout already used elsewhere in this IFID
repository without running DINO twice.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset
from eval_proxy_dinocos import (
    load_dino_encoder,
    preprocess_dinov2_from_m11,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--dataset-ref", default="/video_ssd/lpm/ImageNet/train")
    p.add_argument("--ref-size", type=int, default=50000)
    p.add_argument("--val-size", type=int, default=50000)
    p.add_argument("--resolution", type=int, default=256)

    p.add_argument("--dino-enc-type", default="dinov2-vit-b")
    p.add_argument(
        "--encoder-code-root",
        default="/share/xutongda/REPA-E",
    )

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--out-dir",
        default=(
            "/video_ssd/kongzishang/xutongda/git/IFID/"
            "msac_results/dino_semantics_dinov2b_ref50000_val50000"
        ),
    )
    return p.parse_args()


@torch.inference_mode()
def extract_global_semantics(encoder, x_m11):
    x = preprocess_dinov2_from_m11(x_m11)
    feats = encoder.forward_features(x)

    if not isinstance(feats, dict):
        raise TypeError(
            "Expected repository DINOv2 forward_features() to return dict, "
            f"got {type(feats)}"
        )
    if "x_norm_patchtokens" not in feats:
        raise KeyError(
            "DINO features missing x_norm_patchtokens; keys="
            + str(list(feats.keys()))
        )
    if "x_norm_clstoken" not in feats:
        raise KeyError(
            "DINO features missing x_norm_clstoken; keys="
            + str(list(feats.keys()))
        )

    cls = feats["x_norm_clstoken"].float()
    patch = feats["x_norm_patchtokens"].float()

    if cls.ndim != 2:
        raise RuntimeError(f"Expected CLS [B,C], got {tuple(cls.shape)}")
    if patch.ndim != 3:
        raise RuntimeError(
            f"Expected patch tokens [B,N,C], got {tuple(patch.shape)}"
        )

    meanpatch = patch.mean(dim=1)

    if cls.shape[0] != meanpatch.shape[0]:
        raise RuntimeError("CLS/meanpatch batch mismatch")
    if cls.shape[-1] != meanpatch.shape[-1]:
        raise RuntimeError(
            f"CLS/meanpatch dim mismatch: {cls.shape} vs {meanpatch.shape}"
        )

    # IMPORTANT: no F.normalize() here.
    return cls, meanpatch


def write_names(path, names):
    Path(path).write_text(
        "".join(str(x).replace("\n", " ") + "\n" for x in names),
        encoding="utf-8",
    )


@torch.inference_mode()
def cache_split(
    split_name,
    dataset,
    loader,
    encoder,
    device,
    out_dir,
):
    cls_path = out_dir / f"{split_name}_cls.npy"
    mean_path = out_dir / f"{split_name}_meanpatch.npy"
    names_path = out_dir / f"{split_name}_names.txt"

    cls_mm = None
    mean_mm = None
    names_all = []
    offset = 0
    feat_dim = None

    for img, _y, names in tqdm(loader, desc=f"DINO semantics {split_name}"):
        img = img.to(device, non_blocking=True)
        cls, meanpatch = extract_global_semantics(encoder, img)

        B, C = cls.shape
        if cls_mm is None:
            feat_dim = int(C)
            cls_mm = np.lib.format.open_memmap(
                cls_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(dataset), feat_dim),
            )
            mean_mm = np.lib.format.open_memmap(
                mean_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(dataset), feat_dim),
            )

        if C != feat_dim:
            raise RuntimeError("DINO feature dim changed across batches")

        cls_mm[offset:offset+B] = cls.cpu().numpy()
        mean_mm[offset:offset+B] = meanpatch.cpu().numpy()
        names_all.extend(str(x) for x in names)
        offset += B

        del img, cls, meanpatch

    if offset != len(dataset):
        raise RuntimeError(
            f"{split_name}: cached {offset}, expected {len(dataset)}"
        )
    if len(names_all) != len(dataset):
        raise RuntimeError("name count mismatch")

    cls_mm.flush()
    mean_mm.flush()
    write_names(names_path, names_all)

    # Useful sanity only; do not normalize.
    cls_arr = np.load(cls_path, mmap_mode="r")
    mean_arr = np.load(mean_path, mmap_mode="r")
    stats = {
        "count": int(offset),
        "dim": int(feat_dim),
        "cls_norm_mean": float(
            np.linalg.norm(cls_arr, axis=1).mean()
        ),
        "meanpatch_norm_mean": float(
            np.linalg.norm(mean_arr, axis=1).mean()
        ),
    }
    return stats


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"

    expected = [
        out / "ref_cls.npy",
        out / "ref_meanpatch.npy",
        out / "ref_names.txt",
        out / "val_cls.npy",
        out / "val_meanpatch.npy",
        out / "val_names.txt",
        meta_path,
    ]
    if all(p.exists() for p in expected) and not args.force:
        meta = json.loads(meta_path.read_text())
        if (
            int(meta.get("ref_size", -1)) == args.ref_size
            and int(meta.get("val_size", -1)) == args.val_size
            and int(meta.get("resolution", -1)) == args.resolution
            and meta.get("dino_enc_type") == args.dino_enc_type
        ):
            print(f"[cache] reuse complete semantic cache: {out}")
            return

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
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

    dino, dino_type, dino_arch, loader_module = load_dino_encoder(
        args.dino_enc_type,
        device,
        args.resolution,
        args.encoder_code_root,
    )

    print("=" * 88)
    print("M_SAC DINO semantic cache")
    print("DINO requested      :", args.dino_enc_type)
    print("DINO encoder type   :", dino_type)
    print("DINO architecture   :", dino_arch)
    print("load_encoders module:", loader_module)
    print("features            : x_norm_clstoken + mean(x_norm_patchtokens)")
    print("unit L2 normalize   : NO")
    print("ref / val           :", args.ref_size, "/", args.val_size)
    print("out                 :", out)
    print("=" * 88)

    ref_stats = cache_split(
        "ref", ref_set, ref_loader, dino, device, out
    )
    val_stats = cache_split(
        "val", val_set, val_loader, dino, device, out
    )

    meta = {
        "dataset": args.dataset,
        "dataset_ref": args.dataset_ref,
        "ref_size": int(args.ref_size),
        "val_size": int(args.val_size),
        "resolution": int(args.resolution),
        "dino_enc_type": args.dino_enc_type,
        "dino_encoder_type": dino_type,
        "dino_architecture": dino_arch,
        "load_encoders_module": loader_module,
        "encoder_code_root": str(
            Path(args.encoder_code_root).resolve()
        ),
        "preprocess": (
            "IFID [-1,1] -> [0,1] clamp -> ImageNet mean/std "
            "-> bicubic 256->224, via eval_proxy_dinocos.py"
        ),
        "semantic_features": {
            "cls": "x_norm_clstoken",
            "meanpatch": "mean(x_norm_patchtokens, dim=patch)",
        },
        "post_feature_unit_normalization": False,
        "ref_stats": ref_stats,
        "val_stats": val_stats,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print("\nDONE:", out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(parse_args())
