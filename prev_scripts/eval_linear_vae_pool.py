#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear probing accuracy for VAE latents.

Adapted from facebookresearch/dino/eval_linear.py, but the frozen feature
extractor is an IFID VAE encoder instead of a DINO backbone.

Main protocol:
  image -> VAE.encode(image) -> latent feature -> frozen -> train linear classifier
"""

import argparse
import json
import os
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from omegaconf import OmegaConf

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from ifid.dataset import ImageNetValDataset
from ifid.vae.utils import instantiate_from_config


# ---------------------------------------------------------------------
# VAE / latent helpers
# ---------------------------------------------------------------------
def unwrap_vae_latents(z: Any) -> torch.Tensor:
    """
    Normalize different VAE encode() return types to a Tensor.

    Important:
      torch.Tensor itself has a .mode() method, so Tensor must be returned
      before checking posterior-like .mode().
    """
    if isinstance(z, torch.Tensor):
        return z

    if isinstance(z, (tuple, list)):
        z = z[0]
        if isinstance(z, torch.Tensor):
            return z

    if isinstance(z, dict):
        for key in ["z", "latent", "latents", "sample", "moments"]:
            if key in z:
                z = z[key]
                break
        if isinstance(z, torch.Tensor):
            return z

    if hasattr(z, "mode") and callable(z.mode):
        z = z.mode()
    elif hasattr(z, "sample") and callable(z.sample):
        z = z.sample()

    if isinstance(z, torch.Tensor):
        return z

    raise TypeError(f"Unsupported VAE encode output type: {type(z)}")


@torch.no_grad()
def encode_vae(vae: nn.Module, images: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Robustly call VAE.encode. Some wrappers accept labels, most do not."""
    try:
        if labels is not None:
            z = vae.encode(images, labels)
        else:
            z = vae.encode(images)
    except TypeError:
        z = vae.encode(images)
    return unwrap_vae_latents(z)


def latent_to_feature(
    z: torch.Tensor,
    feature_type: str = "avgpool",
    pool_size: int = 1,
    l2_normalize: bool = False,
) -> torch.Tensor:
    """
    Convert a VAE latent into a single image-level feature vector.

    Main fair protocol, following the spirit of DINO linear eval:
      frozen encoder -> image-level feature vector -> linear classifier

    Supported latent shapes:
      spatial latent: [B, C, H, W]
      token latent:   [B, N, C]
      vector latent:  [B, C]

    feature_type:
      avgpool / gap:
        [B,C,H,W] -> mean over H,W -> [B,C]
        [B,N,C]   -> mean over N   -> [B,C]
        [B,C]     -> [B,C]

      maxpool:
        [B,C,H,W] -> max over H,W -> [B,C]
        [B,N,C]   -> max over N   -> [B,C]
        [B,C]     -> [B,C]

      avgmaxpool:
        concatenate avgpool and maxpool -> [B,2C]
        This is useful as an ablation, but do not mix it with avgpool/maxpool
        in the same main table.

      flatten:
        [B,...] -> [B,prod(...)]
        Kept only for appendix/debug. It is not the recommended fair main metric.

      pool:
        [B,C,H,W] -> adaptive_avg_pool2d(pool_size) -> flatten.
        Kept only for appendix/debug.
    """
    if not isinstance(z, torch.Tensor):
        raise TypeError(type(z))

    z = z.float()

    def _avgpool(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.mean(dim=(2, 3))
        if x.ndim == 3:
            return x.mean(dim=1)  # token latent [B,N,C]
        if x.ndim == 2:
            return x
        raise RuntimeError(f"Unsupported latent shape for avgpool: {tuple(x.shape)}")

    def _maxpool(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.flatten(2).max(dim=2).values
        if x.ndim == 3:
            return x.max(dim=1).values  # token latent [B,N,C]
        if x.ndim == 2:
            return x
        raise RuntimeError(f"Unsupported latent shape for maxpool: {tuple(x.shape)}")

    if feature_type in ("avgpool", "gap"):
        feat = _avgpool(z)
    elif feature_type == "maxpool":
        feat = _maxpool(z)
    elif feature_type == "avgmaxpool":
        feat = torch.cat([_avgpool(z), _maxpool(z)], dim=1)
    elif feature_type == "flatten":
        feat = z.reshape(z.shape[0], -1)
    elif feature_type == "pool":
        if z.ndim != 4:
            raise RuntimeError(f"--feature-type pool only supports [B,C,H,W], got {tuple(z.shape)}")
        feat = F.adaptive_avg_pool2d(z, output_size=(pool_size, pool_size)).reshape(z.shape[0], -1)
    else:
        raise ValueError(feature_type)

    if l2_normalize:
        feat = F.normalize(feat, dim=1)

    return feat.contiguous()


@torch.no_grad()
def infer_feature_dim(
    vae: nn.Module,
    device: torch.device,
    resolution: int,
    feature_type: str,
    pool_size: int,
    l2_normalize: bool,
) -> Tuple[int, Tuple[int, ...]]:
    fake_img = torch.zeros(1, 3, resolution, resolution, device=device)
    fake_y = torch.zeros(1, dtype=torch.long, device=device)
    z = encode_vae(vae, fake_img, fake_y)
    feat = latent_to_feature(z, feature_type=feature_type, pool_size=pool_size, l2_normalize=l2_normalize)
    return feat.shape[1], tuple(z.shape[1:])


# ---------------------------------------------------------------------
# Linear classifier / metrics
# ---------------------------------------------------------------------
class LinearClassifier(nn.Module):
    """Linear layer on top of frozen VAE features, with DINO-style init."""
    def __init__(self, dim: int, num_labels: int = 1000):
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.reshape(x.shape[0], -1))


@torch.no_grad()
def correct_counts(logits: torch.Tensor, target: torch.Tensor, num_labels: int) -> torch.Tensor:
    """Return tensor [correct1, correct5, total]."""
    maxk = 5 if num_labels >= 5 else 1
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    correct1 = correct[:1].reshape(-1).float().sum()
    correct5 = correct[:5].reshape(-1).float().sum() if num_labels >= 5 else correct1
    total = torch.tensor(float(target.numel()), device=logits.device)
    return torch.stack([correct1, correct5, total])


def unpack_batch(batch):
    """Supports ImageNetValDataset -> (img, y, name), or ImageFolder -> (img, y)."""
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise RuntimeError(f"Unsupported batch type: {type(batch)}")


# ---------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------
def train_one_epoch(
    accelerator: Accelerator,
    vae: nn.Module,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    epoch: int,
    args,
) -> Dict[str, float]:
    classifier.train()
    vae.eval()

    loss_sum = torch.tensor(0.0, device=accelerator.device)
    count_sum = torch.tensor(0.0, device=accelerator.device)
    correct_sum = torch.zeros(3, device=accelerator.device)

    progress = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Train epoch {epoch}")

    for batch in progress:
        images, target = unpack_batch(batch)
        images = images.to(accelerator.device, non_blocking=True)
        target = target.to(accelerator.device, non_blocking=True).long()

        with torch.no_grad():
            z = encode_vae(vae, images, target)
            feat = latent_to_feature(
                z,
                feature_type=args.feature_type,
                pool_size=args.pool_size,
                l2_normalize=args.l2_normalize,
            )

        logits = classifier(feat)
        loss = F.cross_entropy(logits, target)

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        optimizer.step()

        bs = target.numel()
        loss_sum += loss.detach() * bs
        count_sum += bs
        correct_sum += correct_counts(logits.detach(), target, args.num_labels)

        if accelerator.is_local_main_process:
            acc1_local = 100.0 * correct_sum[0].item() / max(correct_sum[2].item(), 1.0)
            progress.set_postfix(loss=float(loss.detach().item()), acc1=acc1_local)

    loss_count = torch.stack([loss_sum, count_sum])
    loss_count = accelerator.reduce(loss_count, reduction="sum")
    correct_sum = accelerator.reduce(correct_sum, reduction="sum")

    return {
        "loss": (loss_count[0] / loss_count[1]).item(),
        "acc1": (100.0 * correct_sum[0] / correct_sum[2]).item(),
        "acc5": (100.0 * correct_sum[1] / correct_sum[2]).item(),
    }


@torch.no_grad()
def validate(
    accelerator: Accelerator,
    vae: nn.Module,
    classifier: nn.Module,
    val_loader: DataLoader,
    args,
) -> Dict[str, float]:
    classifier.eval()
    vae.eval()

    loss_sum = torch.tensor(0.0, device=accelerator.device)
    count_sum = torch.tensor(0.0, device=accelerator.device)
    correct_sum = torch.zeros(3, device=accelerator.device)

    progress = tqdm(val_loader, disable=not accelerator.is_local_main_process, desc="Val")

    for batch in progress:
        images, target = unpack_batch(batch)
        images = images.to(accelerator.device, non_blocking=True)
        target = target.to(accelerator.device, non_blocking=True).long()

        z = encode_vae(vae, images, target)
        feat = latent_to_feature(
            z,
            feature_type=args.feature_type,
            pool_size=args.pool_size,
            l2_normalize=args.l2_normalize,
        )

        logits = classifier(feat)
        loss = F.cross_entropy(logits, target)

        bs = target.numel()
        loss_sum += loss.detach() * bs
        count_sum += bs
        correct_sum += correct_counts(logits, target, args.num_labels)

    loss_count = torch.stack([loss_sum, count_sum])
    loss_count = accelerator.reduce(loss_count, reduction="sum")
    correct_sum = accelerator.reduce(correct_sum, reduction="sum")

    return {
        "loss": (loss_count[0] / loss_count[1]).item(),
        "acc1": (100.0 * correct_sum[0] / correct_sum[2]).item(),
        "acc5": (100.0 * correct_sum[1] / correct_sum[2]).item(),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(args):
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to if args.report_to != "none" else None,
        project_config=project_config,
    )

    set_seed(args.seed + accelerator.process_index)
    torch.backends.cudnn.benchmark = True

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    val_transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    if args.no_train_aug:
        train_transform = val_transform
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.resolution, scale=(args.rrc_scale_min, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    train_set = ImageNetValDataset(args.train_data, train_transform, args.train_size)
    val_set = ImageNetValDataset(args.val_data, val_transform, args.val_size)

    per_proc_batch = max(1, args.batch_size // accelerator.num_processes)
    train_loader = DataLoader(
        train_set,
        batch_size=per_proc_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=per_proc_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(accelerator.device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    feature_dim, latent_shape = infer_feature_dim(
        vae=vae,
        device=accelerator.device,
        resolution=args.resolution,
        feature_type=args.feature_type,
        pool_size=args.pool_size,
        l2_normalize=args.l2_normalize,
    )

    if accelerator.is_main_process:
        print("========== Linear Probe VAE ==========")
        print("vae_config:", args.vae_config)
        print("latent_shape:", latent_shape)
        print("feature_type:", args.feature_type)
        print("pool_size:", args.pool_size)
        print("l2_normalize:", args.l2_normalize)
        print("feature_dim:", feature_dim)
        print("num_labels:", args.num_labels)
        print("train images:", len(train_set), "val images:", len(val_set))

    if args.max_feature_dim > 0 and feature_dim > args.max_feature_dim:
        raise RuntimeError(
            f"feature_dim={feature_dim} > max_feature_dim={args.max_feature_dim}. "
            f"Use --feature-type avgpool/maxpool for the main fair protocol, or flatten/pool only for appendix."
        )

    classifier = LinearClassifier(feature_dim, num_labels=args.num_labels).to(accelerator.device)

    actual_lr = args.lr * args.batch_size / 256.0
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=actual_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    classifier, optimizer, train_loader, val_loader = accelerator.prepare(classifier, optimizer, train_loader, val_loader)

    if args.report_to != "none" and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.project_name,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.exp_name}},
        )

    best_acc1 = 0.0
    start_epoch = 0

    ckpt_path = os.path.join(args.output_dir, "checkpoint.pth")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        accelerator.unwrap_model(classifier).load_state_dict(ckpt["classifier"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"])
        best_acc1 = float(ckpt["best_acc1"])
        if accelerator.is_main_process:
            print(f"[Resume] {ckpt_path}, epoch={start_epoch}, best_acc1={best_acc1:.3f}")

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(accelerator, vae, classifier, optimizer, train_loader, epoch, args)
        scheduler.step()

        log_stats = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_stats["loss"],
            "train_acc1": train_stats["acc1"],
            "train_acc5": train_stats["acc5"],
        }

        if (epoch % args.val_freq == 0) or (epoch == args.epochs - 1):
            val_stats = validate(accelerator, vae, classifier, val_loader, args)
            best_acc1 = max(best_acc1, val_stats["acc1"])
            log_stats.update({
                "val_loss": val_stats["loss"],
                "val_acc1": val_stats["acc1"],
                "val_acc5": val_stats["acc5"],
                "best_acc1": best_acc1,
            })

        if accelerator.is_main_process:
            print(json.dumps(log_stats, indent=2))
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            save_dict = {
                "epoch": epoch + 1,
                "classifier": accelerator.unwrap_model(classifier).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc1": best_acc1,
                "feature_dim": feature_dim,
                "latent_shape": latent_shape,
                "args": vars(args),
            }
            torch.save(save_dict, ckpt_path)

        if args.report_to != "none":
            accelerator.log(log_stats, step=epoch)

    if accelerator.is_main_process:
        result = {
            "best_acc1": best_acc1,
            "feature_dim": feature_dim,
            "latent_shape": latent_shape,
            "feature_type": args.feature_type,
            "pool_size": args.pool_size,
            "l2_normalize": args.l2_normalize,
        }
        with open(os.path.join(args.output_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        print("========== Final ==========")
        print(json.dumps(result, indent=2))

    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser("Linear probing on frozen VAE latents")

    parser.add_argument("--vae-config", type=str, required=True)
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--exp-name", type=str, default="linear-probe-vae")
    parser.add_argument("--project-name", type=str, default="vae-linear-probe")

    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=-1)
    parser.add_argument("--val-size", type=int, default=-1)
    parser.add_argument("--num-labels", type=int, default=1000)

    parser.add_argument("--feature-type", type=str, default="avgpool", choices=["avgpool", "maxpool", "avgmaxpool", "gap", "flatten", "pool"])
    parser.add_argument("--pool-size", type=int, default=1)
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--max-feature-dim", type=int, default=-1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val-freq", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256, help="Global batch size.")
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=0.1, help="Base lr before DINO linear scaling rule.")
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report-to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--no-train-aug", action="store_true")
    parser.add_argument("--rrc-scale-min", type=float, default=0.08)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
