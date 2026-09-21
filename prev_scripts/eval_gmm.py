# eval_gmm.py
"""
VAE-only GMM training on REAL latents, using z.

Pipeline:
image -> VAE encoder -> z -> flatten -> optional factor truncate
-> PCA/UMAP/None -> standardize -> train/val split -> train GMM -> BIC

Current interface:
  --vae-config ./configs/FLUX2VAE.yaml

Example:
nohup env CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --main_process_port 29505 eval_gmm.py \
  --base_path /video_ssd/kongzishang/imagenet_val \
  --vae-config ./configs/FLUX2VAE.yaml \
  > /video_ssd/kongzishang/xutongda/git/IFID/losses/eval_gmm_flux2vae.out 2>&1 &
"""

import os
import time
import argparse
import random
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.distributions import MultivariateNormal

from PIL import Image
from torchvision import transforms
from tqdm import tqdm, trange

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import umap
import joblib
import wandb

from accelerate import Accelerator
from accelerate.utils import set_seed


# -------------------------
# config instantiate
# -------------------------
def load_config(path: str):
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f)


def get_obj_from_str(string: str):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Config must contain field `target`.")

    target = config["target"]
    params = config.get("params", {}) or {}
    cls = get_obj_from_str(target)
    return cls(**params)


@torch.no_grad()
def eval_ll_nll(model: nn.Module, data: torch.Tensor, device):
    ll = model(data.to(device)).item()
    nll = -ll
    return ll, nll


# -------------------------
# preprocess
# -------------------------
def preprocess_imgs_vae(imgs: torch.Tensor) -> torch.Tensor:
    # imgs: B,C,H,W in [0,255] -> [-1,1]
    return imgs.float() / 127.5 - 1.0


def _vae_forward_get_posterior_and_z(vae, processed_image: torch.Tensor):
    """
    Current preferred interface:
        Flux2VAE.encode(x) -> z tensor

    Also supports:
        encode(x).latent_dist
        encode(x).sample()
        old forward outputs like (posterior, z)
    """

    # Preferred path for current repo / Flux2VAE.
    if hasattr(vae, "encode") and callable(getattr(vae, "encode")):
        out = vae.encode(processed_image)

        # Flux2VAE path: encode(x) returns z directly.
        if torch.is_tensor(out):
            return None, out

        # diffusers style: encode(x).latent_dist
        if hasattr(out, "latent_dist"):
            posterior = out.latent_dist

            if hasattr(posterior, "mean") and torch.is_tensor(posterior.mean):
                z = posterior.mean
            elif hasattr(posterior, "mode") and callable(getattr(posterior, "mode")):
                z = posterior.mode()
            elif hasattr(posterior, "sample") and callable(getattr(posterior, "sample")):
                z = posterior.sample()
            else:
                raise RuntimeError("Cannot obtain z from out.latent_dist.")

            if not torch.is_tensor(z):
                raise RuntimeError(f"posterior produced non-tensor z: {type(z)}")

            return posterior, z

        # posterior directly
        if hasattr(out, "mean") and torch.is_tensor(out.mean):
            return out, out.mean

        if hasattr(out, "mode") and callable(getattr(out, "mode")):
            z = out.mode()
            if torch.is_tensor(z):
                return out, z

        if hasattr(out, "sample") and callable(getattr(out, "sample")):
            z = out.sample()
            if torch.is_tensor(z):
                return out, z

        # tuple/list from encode
        if isinstance(out, (tuple, list)):
            if len(out) >= 2 and torch.is_tensor(out[1]):
                return out[0], out[1]
            if len(out) >= 1 and torch.is_tensor(out[0]):
                return None, out[0]

        raise RuntimeError(f"vae.encode(x) returned unsupported type: {type(out)}")

    # Fallback: old forward interface.
    try:
        out = vae(processed_image, return_recon=False)
    except TypeError:
        out = vae(processed_image)

    if isinstance(out, (tuple, list)) and len(out) >= 2:
        posterior = out[0]
        z = out[1]

        if not torch.is_tensor(z):
            raise RuntimeError("VAE output[1] is not a Tensor; cannot interpret as z.")

        return posterior, z

    raise RuntimeError("Unexpected VAE output; cannot obtain posterior and z.")


# -------------------------
# latent flatten helper
# -------------------------
def flatten_latent_for_gmm(z: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Convert different VAE latent layouts into a fixed vector per image.

    Supported examples:
      - spatial latent: B,C,H,W
      - token latent  : B,L,C or B,C,L
      - global latent : B,D
      - generic latent: B,...

    GMM only needs one feature vector per image, so there is no need to
    force a BCHW interpretation. This is the key compatibility path for
    1D/token VAEs.
    """
    if not torch.is_tensor(z):
        raise RuntimeError(f"Expected tensor latent z, got {type(z)}")

    if z.dim() == 0:
        raise RuntimeError("Scalar latent is unsupported.")

    if z.dim() == 1:
        # Rare case: either B scalar latents or one sample flattened vector.
        if z.shape[0] == batch_size:
            z = z.view(batch_size, 1)
        elif batch_size == 1:
            z = z.view(1, -1)
        else:
            raise RuntimeError(
                f"Cannot infer batch dimension for 1D latent shape={tuple(z.shape)} "
                f"with batch_size={batch_size}"
            )

    if z.shape[0] != batch_size:
        raise RuntimeError(
            f"Latent batch mismatch: z.shape={tuple(z.shape)}, batch_size={batch_size}. "
            "Expected z.shape[0] to match input batch size."
        )

    return z.detach().float().contiguous().flatten(1).cpu()


def latent_layout_name(z: torch.Tensor) -> str:
    if not torch.is_tensor(z):
        return str(type(z))
    if z.dim() == 4:
        return "BCHW"
    if z.dim() == 3:
        return "BLC_or_BCL_token"
    if z.dim() == 2:
        return "BD_global"
    if z.dim() == 1:
        return "B_or_single_vector"
    return f"B_plus_{z.dim() - 1}D"


# -------------------------
# dataset
# -------------------------
class ImageNetFolderDataset(Dataset):
    """
    ImageNet-style folder:
        root/class_name/*.jpg
    """

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        self.samples = []
        classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())

        if len(classes) == 0:
            raise RuntimeError(f"No class folders found under: {root}")

        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for cls_name in classes:
            cls_dir = os.path.join(root, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(exts):
                    path = os.path.join(cls_dir, fname)
                    label = class_to_idx[cls_name]
                    self.samples.append((path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under: {root}")

        print(f"[Dataset] root       : {root}")
        print(f"[Dataset] num classes: {len(classes)}")
        print(f"[Dataset] num images : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# -------------------------
# GMM
# -------------------------
class GMM(nn.Module):
    def __init__(self, n_components, n_features):
        super(GMM, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.weights = nn.Parameter(torch.ones(n_components) / n_components)
        self.means = nn.Parameter(torch.randn(n_components, n_features))
        self.log_covariances = nn.Parameter(torch.zeros(n_components, n_features))

    def forward(self, x):
        likelihoods = []

        for i in range(self.n_components):
            diag_cov = torch.exp(self.log_covariances[i])
            dist = MultivariateNormal(self.means[i], torch.diag(diag_cov))
            likelihoods.append(dist.log_prob(x))

        likelihoods = torch.stack(likelihoods, dim=-1)

        weighted_log_likelihoods = likelihoods + torch.log(self.weights + 1e-10)
        total_likelihood = torch.logsumexp(weighted_log_likelihoods, dim=-1)

        return total_likelihood.mean()

    def bic(self, x):
        n_samples = x.size(0)
        n_params = self.n_components * (1 + self.n_features + self.n_features)
        log_likelihood = self.forward(x).item() * n_samples
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        return bic


def save_component_statistics(model, n_components, output_path):
    component_stats = []
    min_distance = float("inf")

    for i in range(n_components):
        for j in range(i + 1, n_components):
            distance = torch.norm(model.means[i] - model.means[j]).item()
            min_distance = min(min_distance, distance)

    for i in range(n_components):
        mean_norm = torch.norm(model.means[i]).item()
        cov_norm = torch.norm(torch.exp(model.log_covariances[i])).item()

        mu = model.means[i].detach().cpu().numpy()
        cov_matrix = torch.exp(model.log_covariances[i]).detach().cpu().numpy()

        if cov_matrix.ndim == 1:
            e_x2 = np.linalg.norm(mu) ** 2 + np.sum(cov_matrix)
        else:
            e_x2 = np.linalg.norm(mu) ** 2 + np.trace(cov_matrix)

        component_stats.append(
            [
                n_components,
                i + 1,
                mean_norm,
                cov_norm,
                e_x2,
                min_distance,
            ]
        )

    df = pd.DataFrame(
        component_stats,
        columns=[
            "n_components",
            "cluster",
            "mean_norm",
            "cov_norm",
            "second_moment",
            "min_mean_distance",
        ],
    )

    if os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
        print(f"Statistics appended to {output_path}")
    else:
        df.to_csv(output_path, index=False)
        print(f"Statistics saved to {output_path}")


# -------------------------
# VAE load
# -------------------------
def load_vae(args, device, accelerator=None):
    cfg = load_config(args.vae_config)

    if (accelerator is None) or accelerator.is_main_process:
        print(f"[VAE] config: {args.vae_config}")
        print(f"[VAE] target: {cfg.get('target')}")

    vae = instantiate_from_config(cfg)
    vae = vae.to(device)
    vae.eval()

    for p in vae.parameters():
        p.requires_grad_(False)

    return vae


# -------------------------
# latents collect
# -------------------------
def collect_real_latents(image_root, vae, args, device, accelerator: Accelerator):
    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    ds = ImageNetFolderDataset(root=image_root, transform=transform)

    loader = DataLoader(
        ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    feats = []
    n_collected = 0

    pbar = tqdm(
        loader,
        desc="Collect REAL latents",
        disable=not accelerator.is_local_main_process,
    )

    for img01, _ in pbar:
        img01 = img01.to(device, non_blocking=True)
        raw255 = img01 * 255.0

        with torch.no_grad():
            processed = preprocess_imgs_vae(raw255)
            _, z = _vae_forward_get_posterior_and_z(vae, processed)

            flat = flatten_latent_for_gmm(z, batch_size=img01.size(0))

            if n_collected == 0 and accelerator.is_main_process:
                print(f"[Latent] raw z shape : {tuple(z.shape)}")
                print(f"[Latent] layout      : {latent_layout_name(z)}")
                print(f"[Latent] flat shape  : {tuple(flat.shape)}")

        feats.append(flat)
        n_collected += flat.size(0)

        pbar.set_postfix({"n": n_collected, "dim": flat.size(1)})

        if args.num_real > 0 and n_collected >= args.num_real:
            break

    x = torch.cat(feats, dim=0)

    if args.num_real > 0:
        x = x[: args.num_real]

    return x


def apply_factor_truncate(x_cpu: torch.Tensor, factor: float):
    d0 = x_cpu.size(1)

    if factor < 0:
        d = int(d0 * abs(factor))
    else:
        d = int(d0 * factor)

    d = max(1, min(d, d0))
    return x_cpu[:, :d], d0, d


# -------------------------
# train / metric
# -------------------------
def maybe_wandb_log(payload):
    if wandb.run is not None:
        wandb.log(payload)


def train_gmm(train_data, val_data, n_components, batch_size, n_iter, lr, device, args, num_workers=4):
    n_features = train_data.shape[1]

    model = GMM(n_components, n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_dataset = TensorDataset(val_data)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    train_losses = []
    val_losses = []

    for local_epoch in trange(n_iter, desc=f"GMM k={n_components}"):
        model.train()
        batch_train_losses = []

        for batch_data in train_dataloader:
            batch = batch_data[0].to(device)

            optimizer.zero_grad(set_to_none=True)

            loss = -model(batch)
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())

        epoch_train_loss = float(np.mean(batch_train_losses))
        train_losses.append(epoch_train_loss)

        model.eval()
        batch_val_losses = []

        with torch.no_grad():
            for batch_data in val_dataloader:
                batch = batch_data[0].to(device)
                loss = -model(batch)
                batch_val_losses.append(loss.item())

        epoch_val_loss = float(np.mean(batch_val_losses))
        val_losses.append(epoch_val_loss)

        maybe_wandb_log(
            {
                f"train_loss_n_components_{n_components}": epoch_train_loss,
                f"val_loss_n_components_{n_components}": epoch_val_loss,
                "epoch": local_epoch,
                "n_components": n_components,
            }
        )

    return model, train_losses, val_losses


def calculate_metric(train_data, val_data, args, device="cpu"):
    bics = []
    final_train_losses = []
    final_val_losses = []

    for n_components in args.components_list:
        maybe_wandb_log({"current_n_components": n_components})

        model, train_losses, val_losses = train_gmm(
            train_data=train_data,
            val_data=val_data,
            n_components=n_components,
            batch_size=args.batch_size,
            n_iter=args.n_iter,
            lr=args.lr,
            device=device,
            args=args,
        )

        bic = model.bic(train_data.to(device))
        bics.append(bic)

        maybe_wandb_log({"BIC": bic, "n_components": n_components})

        final_train_losses.append(train_losses[-1])
        final_val_losses.append(val_losses[-1])

    return bics, final_train_losses, final_val_losses


def log_final_loss_and_bic_plot(bics, final_train_losses, final_val_losses, components_list):
    plt.figure(figsize=(8, 6))

    plt.plot(components_list, final_train_losses, label="Final Train Loss Curve", marker="o", linestyle="-")
    plt.plot(components_list, final_val_losses, label="Final Val Loss Curve", marker="o", linestyle="-")
    plt.plot(components_list, bics, label="BIC Curve", marker="o", linestyle="--")

    plt.xlabel("Number of Components")
    plt.ylabel("Value")
    plt.title("Final Train/Val Loss and BIC Curves")
    plt.legend()
    plt.grid(True)

    if wandb.run is not None:
        wandb.log({"Final_Train_Val_Loss_and_BIC_Plot": wandb.Image(plt)})

    plt.close()


# -------------------------
# argparse + main
# -------------------------
def build_parser():
    p = argparse.ArgumentParser("VAE-only GMM eval, config-based VAE interface.")

    p.add_argument("--base_path", "--base-path", dest="base_path", type=str, required=True)
    p.add_argument("--exp", type=str, default="flux2vae_gmm")
    p.add_argument("--out_dir", "--out-dir", dest="out_dir", type=str, default="./losses/gmm")

    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=8)

    # VAE config
    p.add_argument("--vae-config", type=str, required=True)

    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--num_real", "--num-real", dest="num_real", type=int, default=-1, help="<=0 means use all")

    # reduction
    p.add_argument("--dim_reduction", "--dim-reduction", dest="dim_reduction", type=str, default="UMAP", choices=["None", "PCA", "UMAP"])
    p.add_argument("--target_dim", "--target-dim", dest="target_dim", type=int, default=100)
    p.add_argument("--factor", type=float, default=1.0)

    # GMM
    p.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--n_iter", "--n-iter", dest="n_iter", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_components", "--min-components", dest="min_components", type=int, default=2)
    p.add_argument("--max_components", "--max-components", dest="max_components", type=int, default=20)
    p.add_argument("--components_list", "--components-list", dest="components_list", type=int, nargs="+", default=[100])

    # misc
    p.add_argument("--seed", type=int, default=0)

    # wandb
    p.add_argument("--use_wandb", "--use-wandb", dest="use_wandb", type=int, default=1)
    p.add_argument("--wandb_project", "--wandb-project", dest="wandb_project", type=str, default="GMM-Training")

    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.num_processes != 1:
        raise RuntimeError("eval_gmm.py is intended for --num_processes=1 only.")

    if accelerator.is_main_process:
        print("[Device]", device)

    set_seed(args.seed + accelerator.process_index)
    random.seed(args.seed + accelerator.process_index)
    np.random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)

    # VAE
    vae = load_vae(args, device, accelerator)

    # collect REAL latents
    t0 = time.time()
    real_x = collect_real_latents(args.base_path, vae, args, device, accelerator)

    if accelerator.is_main_process:
        print("[Stage] real done", tuple(real_x.shape), "time:", time.time() - t0)

    # factor truncate
    real_x, d0, d = apply_factor_truncate(real_x, args.factor)

    if accelerator.is_main_process:
        print(f"[Latents] original_dim={d0} -> latent_dim={d}")

    # reduction
    reducer = None

    if args.dim_reduction == "PCA":
        reducer = PCA(n_components=args.target_dim)
        real_red = torch.tensor(reducer.fit_transform(real_x.numpy()), dtype=torch.float32)

    elif args.dim_reduction == "UMAP":
        reducer = umap.UMAP(n_components=args.target_dim)
        real_red = torch.tensor(reducer.fit_transform(real_x.numpy()), dtype=torch.float32)

    else:
        real_red = real_x.float()

    # standardize
    mean = real_red.mean(dim=0, keepdim=True)
    std = real_red.std(dim=0, keepdim=True)
    real_std = (real_red - mean) / (std + 1e-8)

    # split
    train_np, val_np = train_test_split(
        real_std.numpy(),
        test_size=0.2,
        random_state=42,
    )

    train_data = torch.tensor(train_np, dtype=torch.float32)
    val_data = torch.tensor(val_np, dtype=torch.float32)

    if accelerator.is_main_process:
        print("[Split] train", tuple(train_data.shape), "val", tuple(val_data.shape))

    # wandb
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.exp}",
            config=vars(args),
        )

    # train + BIC
    bics, final_train_losses, final_val_losses = calculate_metric(
        train_data=train_data,
        val_data=val_data,
        args=args,
        device=device,
    )

    if accelerator.is_main_process:
        print("\n================ FINAL GMM LOSSES ================")

        for k, tr, va, bic in zip(args.components_list, final_train_losses, final_val_losses, bics):
            print(f"k={k:4d} | final_train_loss={tr:.6f} | final_val_loss={va:.6f} | BIC={bic:.6f}")

        print("final_train_losses =", final_train_losses)
        print("final_val_losses   =", final_val_losses)
        print("bics               =", bics)
        print("===============================================\n")

        log_final_loss_and_bic_plot(
            bics=bics,
            final_train_losses=final_train_losses,
            final_val_losses=final_val_losses,
            components_list=args.components_list,
        )

        if wandb.run is not None:
            wandb.finish()

        print("GMM evaluation complete!")


if __name__ == "__main__":
    main()