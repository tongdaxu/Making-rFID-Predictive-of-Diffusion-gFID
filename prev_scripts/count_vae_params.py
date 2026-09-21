# import argparse
# from pathlib import Path

# import torch
# from omegaconf import OmegaConf

# from ifid.vae.utils import instantiate_from_config


# VAE_CONFIGS = [
#     ("SD-VAE", "SDVAE.yaml"),
#     ("SD3-VAE", "SD3VAE.yaml"),
#     ("FLUX-VAE", "FLUXVAE.yaml"),
#     ("DC-AE", "DCAE.yaml"),
#     ("SOFT-VQ", "SOFTVQ.yaml"),
#     ("VA-VAE", "VAVAE.yaml"),
#     ("VA-VAE-64", "VAVAE64.yaml"),
#     ("EQ-VAE", "EQVAE.yaml"),
#     ("MAE-TOK", "MAETOK.yaml"),
#     ("IN-VAE", "INVAE.yaml"),
#     ("REPAE-SDVAE", "REPAEVAE.yaml"),
#     ("DE-TOK", "DETOK.yaml"),
#     ("QwenImg-VAE", "QWVAE.yaml"),
#     ("RAE", "RAE.yaml"),
#     ("SVG", "SVG.yaml"),
#     ("FLUX2-VAE", "FLUX2VAE.yaml"),
#     ("SVG T2I", "SVGT2I_P_STAGE1_256.yaml"),
#     ("DM-VAE", "DMVAE.yaml"),
#     ("UAE", "UAE.yaml"),
#     ("VTP-S", "VTPS.yaml"),
#     ("VTP-B", "VTPB.yaml"),
#     ("VTP-L", "VTPL.yaml"),
#     ("RAE T2I SigLIP-2", "ScaleRAE_SIGLIP2.yaml"),
#     ("RAE T2I WebSSL", "ScaleRAE_WEBSSL.yaml"),
#     ("PAE DINOv2", "PAE_DINOv2L_d32.yaml"),
# ]


# def count_params(model):
#     return sum(p.numel() for p in model.parameters())


# def count_trainable_params(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def format_params(n):
#     return f"{n:,} ({n / 1e6:.3f}M)"


# def main(args):
#     config_dir = Path(args.config_dir)

#     print(f"{'Name':<22} {'Config':<32} {'Total Params':>24} {'Trainable Params':>24}")
#     print("-" * 110)

#     for name, cfg_name in VAE_CONFIGS:
#         cfg_path = config_dir / cfg_name

#         if not cfg_path.exists():
#             print(f"{name:<22} {cfg_name:<32} {'MISSING CONFIG':>24} {'-':>24}")
#             continue

#         try:
#             with torch.no_grad():
#                 vae_config = OmegaConf.load(cfg_path)
#                 vae = instantiate_from_config(vae_config)

#                 total_params = count_params(vae)
#                 trainable_params = count_trainable_params(vae)

#                 print(
#                     f"{name:<22} "
#                     f"{cfg_name:<32} "
#                     f"{format_params(total_params):>24} "
#                     f"{format_params(trainable_params):>24}"
#                 )

#                 del vae

#         except Exception as e:
#             print(f"{name:<22} {cfg_name:<32} {'ERROR':>24} {str(e)[:24]:>24}")

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--config-dir",
#         type=str,
#         default="./configs",
#         help="Directory containing VAE yaml config files.",
#     )
#     args = parser.parse_args()
#     main(args)


import argparse
import importlib
import os
import sys
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf

# ===== 关键：把项目根目录和可能包含 cont/ 的目录加入 sys.path =====
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent

CANDIDATE_PYTHONPATHS = [
    REPO_ROOT,
    REPO_ROOT / "ifid" / "vae",
    REPO_ROOT / "ifid",
    REPO_ROOT / "vae",
]

for p in CANDIDATE_PYTHONPATHS:
    if p.exists():
        sys.path.insert(0, str(p))

from ifid.vae.utils import instantiate_from_config


VAE_CONFIGS = [
    ("SD-VAE", "SDVAE.yaml"),
    ("SD3-VAE", "SD3VAE.yaml"),
    ("FLUX-VAE", "FLUXVAE.yaml"),
    ("DC-AE", "DCAE.yaml"),
    ("SOFT-VQ", "SOFTVQ.yaml"),
    ("VA-VAE", "VAVAE.yaml"),
    ("VA-VAE-64", "VAVAE64.yaml"),
    ("EQ-VAE", "EQVAE.yaml"),
    ("MAE-TOK", "MAETOK.yaml"),
    ("IN-VAE", "INVAE.yaml"),
    ("REPAE-SDVAE", "REPAEVAE.yaml"),
    ("DE-TOK", "DETOK.yaml"),
    ("QwenImg-VAE", "QWVAE.yaml"),
    ("RAE", "RAE.yaml"),
    ("SVG", "SVG.yaml"),
    ("FLUX2-VAE", "FLUX2VAE.yaml"),
    ("SVG T2I", "SVGT2I_P_STAGE1_256.yaml"),
    ("DM-VAE", "DMVAE.yaml"),
    ("UAE", "UAE.yaml"),
    ("VTP-S", "VTPS.yaml"),
    ("VTP-B", "VTPB.yaml"),
    ("VTP-L", "VTPL.yaml"),
    ("RAE T2I SigLIP-2", "ScaleRAE_SIGLIP2.yaml"),
    ("RAE T2I WebSSL", "ScaleRAE_WEBSSL.yaml"),
    ("PAE DINOv2", "PAE_DINOv2L_d32.yaml"),
]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(n):
    return f"{n:,} ({n / 1e6:.3f}M)"


def collect_targets(obj):
    """
    递归收集 YAML 里所有 target 字段，方便定位哪个模块 import 失败。
    """
    targets = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "target" and isinstance(v, str):
                targets.append(v)
            else:
                targets.extend(collect_targets(v))
    elif isinstance(obj, list):
        for item in obj:
            targets.extend(collect_targets(item))

    return targets


def check_target_importable(target):
    """
    检查 target 对应的 Python module 能不能 import。
    """
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    getattr(module, class_name)


def main(args):
    config_dir = Path(args.config_dir)

    print("Python executable:", sys.executable)
    print("Working directory:", os.getcwd())
    print("Extra sys.path:")
    for p in CANDIDATE_PYTHONPATHS:
        if p.exists():
            print("  ", p)
    print()

    print(f"{'Name':<22} {'Config':<32} {'Total Params':>24} {'Trainable Params':>24}")
    print("-" * 110)

    failed = []

    for name, cfg_name in VAE_CONFIGS:
        cfg_path = config_dir / cfg_name

        if not cfg_path.exists():
            print(f"{name:<22} {cfg_name:<32} {'MISSING CONFIG':>24} {'-':>24}")
            failed.append((name, cfg_name, "missing config"))
            continue

        try:
            vae_config = OmegaConf.load(cfg_path)
            cfg_dict = OmegaConf.to_container(vae_config, resolve=False)

            targets = collect_targets(cfg_dict)

            if args.print_targets:
                print(f"\n[{name}] targets:")
                for t in targets:
                    print("  ", t)

            # 先检查 target import，报错会更清楚
            for t in targets:
                check_target_importable(t)

            with torch.no_grad():
                vae = instantiate_from_config(vae_config)

            total_params = count_params(vae)
            trainable_params = count_trainable_params(vae)

            print(
                f"{name:<22} "
                f"{cfg_name:<32} "
                f"{format_params(total_params):>24} "
                f"{format_params(trainable_params):>24}"
            )

            del vae

        except Exception as e:
            print(f"{name:<22} {cfg_name:<32} {'ERROR':>24} {str(e)[:24]:>24}")
            failed.append((name, cfg_name, str(e)))

            if args.verbose:
                print("\n" + "=" * 80)
                print(f"FAILED CONFIG: {name} / {cfg_name}")
                print(f"CONFIG PATH: {cfg_path}")
                print("ERROR:")
                traceback.print_exc()
                print("=" * 80 + "\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if failed:
        print("\nFailed configs:")
        for name, cfg_name, err in failed:
            print(f"  - {name:<22} {cfg_name:<32} {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, default="./configs")
    parser.add_argument("--print-targets", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)