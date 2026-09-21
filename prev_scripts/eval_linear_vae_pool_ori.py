#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINO-style linear probing on frozen IFID VAE latents.

This keeps the old config-based IFID interface:
  image -> VAE.encode(image) -> latent -> avgpool -> linear classifier

It intentionally does not import MiniMax-AI/VTP. Use this script when your
models are loaded by IFID yaml configs rather than VTP HuggingFace directories.
"""

import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)


def unwrap_vae_latents(z: Any) -> torch.Tensor:
    if isinstance(z, torch.Tensor):
        return z

    if isinstance(z, (tuple, list)):
        z = z[0]
        if isinstance(z, torch.Tensor):
            return z

    if isinstance(z, dict):
        for key in ("z", "latent", "latents", "sample", "moments"):
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
    z = z.float()

    def avgpool(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.mean(dim=(2, 3))
        if x.ndim == 3:
            return x.mean(dim=1)
        if x.ndim == 2:
            return x
        raise RuntimeError(f"Unsupported latent shape for avgpool: {tuple(x.shape)}")

    def maxpool(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.flatten(2).max(dim=2).values
        if x.ndim == 3:
            return x.max(dim=1).values
        if x.ndim == 2:
            return x
        raise RuntimeError(f"Unsupported latent shape for maxpool: {tuple(x.shape)}")

    if feature_type in ("avgpool", "gap"):
        feat = avgpool(z)
    elif feature_type == "maxpool":
        feat = maxpool(z)
    elif feature_type == "avgmaxpool":
        feat = torch.cat([avgpool(z), maxpool(z)], dim=1)
    elif feature_type == "flatten":
        feat = z.reshape(z.shape[0], -1)
    elif feature_type == "pool":
        if z.ndim != 4:
            raise RuntimeError(f"--feature-type pool requires [B,C,H,W], got {tuple(z.shape)}")
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
    feat = latent_to_feature(z, feature_type, pool_size, l2_normalize)
    return feat.shape[1], tuple(z.shape[1:])


class LinearClassifier(nn.Module):
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
    maxk = 5 if num_labels >= 5 else 1
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    correct1 = correct[:1].reshape(-1).float().sum()
    correct5 = correct[:5].reshape(-1).float().sum() if num_labels >= 5 else correct1
    total = torch.tensor(float(target.numel()), device=logits.device)
    return torch.stack([correct1, correct5, total])


def unpack_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise RuntimeError(f"Unsupported batch type: {type(batch)}")


def has_imagefolder_classes(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return any(os.path.isdir(os.path.join(path, item)) for item in os.listdir(path))


def maybe_limit_dataset(dataset, size: int):
    if size is None or size < 0 or size >= len(dataset):
        return dataset
    return Subset(dataset, range(size))


def build_dataset(path: str, transform, size: int, backend: str):
    if backend not in ("auto", "imagefolder", "ifid"):
        raise ValueError(f"Unknown dataset backend: {backend}")

    if backend == "imagefolder" or (backend == "auto" and has_imagefolder_classes(path)):
        dataset = ImageFolder(path, transform=transform)
    else:
        from ifid.dataset import ImageNetValDataset
        dataset = ImageNetValDataset(path, transform, size)
        size = -1

    return maybe_limit_dataset(dataset, size)


def make_transforms(args):
    if args.input_norm == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif args.input_norm == "half":
        mean, std = HALF_MEAN, HALF_STD
    else:
        raise ValueError(args.input_norm)

    resize_size = args.resize_size if args.resize_size > 0 else args.resolution
    val_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if args.no_train_aug:
        train_transform = val_transform
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                args.resolution,
                scale=(args.rrc_scale_min, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return train_transform, val_transform


def load_ifid_vae(config_path: str, device: torch.device) -> nn.Module:
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    vae_config = OmegaConf.load(config_path)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


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
            feat = latent_to_feature(z, args.feature_type, args.pool_size, args.l2_normalize)

        logits = classifier(feat)
        loss = F.cross_entropy(logits, target)

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        optimizer.step()

        bs = target.numel()
        loss_sum += loss.detach() * bs
        count_sum += bs
        correct_sum += correct_counts(logits.detach(), target, args.num_labels)

    loss_count = accelerator.reduce(torch.stack([loss_sum, count_sum]), reduction="sum")
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
        feat = latent_to_feature(z, args.feature_type, args.pool_size, args.l2_normalize)
        logits = classifier(feat)
        loss = F.cross_entropy(logits, target)

        bs = target.numel()
        loss_sum += loss.detach() * bs
        count_sum += bs
        correct_sum += correct_counts(logits, target, args.num_labels)

    loss_count = accelerator.reduce(torch.stack([loss_sum, count_sum]), reduction="sum")
    correct_sum = accelerator.reduce(correct_sum, reduction="sum")
    return {
        "loss": (loss_count[0] / loss_count[1]).item(),
        "acc1": (100.0 * correct_sum[0] / correct_sum[2]).item(),
        "acc5": (100.0 * correct_sum[1] / correct_sum[2]).item(),
    }


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

    train_transform, val_transform = make_transforms(args)
    train_set = build_dataset(args.train_data, train_transform, args.train_size, args.dataset_backend)
    val_set = build_dataset(args.val_data, val_transform, args.val_size, args.dataset_backend)

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

    vae = load_ifid_vae(args.vae_config, accelerator.device)
    feature_dim, latent_shape = infer_feature_dim(
        vae, accelerator.device, args.resolution, args.feature_type, args.pool_size, args.l2_normalize
    )

    if accelerator.is_main_process:
        print("========== Linear Probe IFID VAE ==========")
        print("vae_config:", args.vae_config)
        print("latent_shape:", latent_shape)
        print("feature_type:", args.feature_type)
        print("feature_dim:", feature_dim)
        print("input_norm:", args.input_norm)
        print("train images:", len(train_set), "val images:", len(val_set))

    if args.max_feature_dim > 0 and feature_dim > args.max_feature_dim:
        raise RuntimeError(f"feature_dim={feature_dim} > max_feature_dim={args.max_feature_dim}")

    classifier = LinearClassifier(feature_dim, num_labels=args.num_labels).to(accelerator.device)
    actual_lr = args.lr * args.batch_size / 256.0
    optimizer = torch.optim.SGD(classifier.parameters(), lr=actual_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    classifier, optimizer, train_loader, val_loader = accelerator.prepare(classifier, optimizer, train_loader, val_loader)

    if args.report_to != "none" and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.project_name,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.exp_name}},
        )

    best_acc1 = 0.0
    ckpt_path = os.path.join(args.output_dir, "checkpoint.pth")
    start_epoch = 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        accelerator.unwrap_model(classifier).load_state_dict(ckpt["classifier"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"])
        best_acc1 = float(ckpt["best_acc1"])

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
            torch.save(
                {
                    "epoch": epoch + 1,
                    "classifier": accelerator.unwrap_model(classifier).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc1": best_acc1,
                    "feature_dim": feature_dim,
                    "latent_shape": latent_shape,
                    "args": vars(args),
                },
                ckpt_path,
            )

        if args.report_to != "none":
            accelerator.log(log_stats, step=epoch)

    if accelerator.is_main_process:
        result = {
            "best_acc1": best_acc1,
            "feature_dim": feature_dim,
            "latent_shape": latent_shape,
            "feature_type": args.feature_type,
            "input_norm": args.input_norm,
            "protocol": "IFID VAE encode + latent pooling + DINO-style linear classifier",
        }
        with open(os.path.join(args.output_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        print("========== Final ==========")
        print(json.dumps(result, indent=2))

    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser("Linear probing on frozen IFID VAE latents")
    parser.add_argument("--vae-config", type=str, required=True)
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--exp-name", type=str, default="linear-probe-vae")
    parser.add_argument("--project-name", type=str, default="vae-linear-probe")

    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize-size", type=int, default=-1)
    parser.add_argument("--input-norm", type=str, default="half", choices=["half", "imagenet"])
    parser.add_argument("--dataset-backend", type=str, default="auto", choices=["auto", "imagefolder", "ifid"])
    parser.add_argument("--train-size", type=int, default=-1)
    parser.add_argument("--val-size", type=int, default=-1)
    parser.add_argument("--num-labels", type=int, default=1000)

    parser.add_argument("--feature-type", type=str, default="avgpool", choices=["avgpool", "maxpool", "avgmaxpool", "gap", "flatten", "pool"])
    parser.add_argument("--pool-size", type=int, default=1)
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--max-feature-dim", type=int, default=-1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val-freq", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
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
    main(parse_args())
