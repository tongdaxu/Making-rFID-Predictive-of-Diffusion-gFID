#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINO/VTP-style linear probing for VTP HuggingFace models.

This follows the public VTP linear probing implementation:
  MiniMax-AI/VTP/tools/test_linear_probing_hf.py

Important protocol detail:
  - This probes SSL/ViT features from get_intermediate_layers_feature().
  - It does not probe reconstruction latents from get_reconstruction_latents().
  - "avgpool" here means concatenating the average pooled patch tokens, as in
    DINO/VTP linear eval, not globally averaging the VAE bottleneck latent.
"""

import argparse
import json
import logging
import os
import random
from typing import Dict, List, Sequence, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from vtp.models.vtp_hf import VTPModel


LOGGER = logging.getLogger("linear_probe_vtp_original")

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
CROP_SIZE = 224
RESIZE_SIZE = 256

DEFAULT_LEARNING_RATES = (
    1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4,
    1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1,
)


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def setup_logging() -> None:
    level = logging.INFO if is_main_process() else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def seed_everything(seed: int) -> None:
    seed = seed + get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_train_transform(crop_size: int = CROP_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def make_eval_transform(resize_size: int = RESIZE_SIZE, crop_size: int = CROP_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def init_device(use_ddp: bool, device_arg: str) -> torch.device:
    if use_ddp:
        if not is_dist_avail_and_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device(device_arg)


def precision_to_dtype(precision: str) -> torch.dtype:
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    if precision in ("fp16", "float16"):
        return torch.float16
    return torch.float32


class FeatureExtractor(nn.Module):
    """Frozen VTP feature extractor used by the original VTP linear probe."""

    def __init__(self, model: VTPModel, n_last_blocks: int, autocast_dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_dtype = autocast_dtype

    def forward(self, images: torch.Tensor):
        device_type = "cuda" if images.is_cuda else "cpu"
        use_autocast = images.is_cuda and self.autocast_dtype != torch.float32
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device_type, dtype=self.autocast_dtype, enabled=use_autocast):
                return self.model.get_intermediate_layers_feature(
                    images, n=self.n_last_blocks, return_class_token=True
                )


def create_linear_input(x_tokens_list, use_n_blocks: int, use_avgpool: bool) -> torch.Tensor:
    """Create DINO/VTP linear classifier input from intermediate features."""
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)

    if use_avgpool:
        patch_avg = torch.mean(intermediate_output[-1][0], dim=1)
        output = torch.cat((output, patch_avg), dim=-1)

    return output.float().reshape(output.shape[0], -1)


class LinearClassifier(nn.Module):
    """Linear layer on top of frozen features, with DINO-style init."""

    def __init__(self, out_dim: int, use_n_blocks: int, use_avgpool: bool, num_classes: int = 1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


class AllClassifiers(nn.Module):
    """Container for the official LR/block sweep."""

    def __init__(self, classifiers_dict: Dict[str, nn.Module]):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict(classifiers_dict)

    def forward(self, inputs) -> Dict[str, torch.Tensor]:
        return {k: v(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self) -> int:
        return len(self.classifiers_dict)


class InfiniteSampler(Sampler):
    """Wrap another sampler to yield an infinite stream."""

    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.epoch = 0

    def __iter__(self):
        while True:
            if hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(self.epoch)
            yield from iter(self.sampler)
            self.epoch += 1

    def __len__(self) -> int:
        return int(1e18)


def scale_lr(learning_rate: float, batch_size_per_gpu: int) -> float:
    return learning_rate * (batch_size_per_gpu * get_world_size()) / 256.0


def setup_linear_classifiers(
    sample_output,
    n_last_blocks_list: Sequence[int],
    learning_rates: Sequence[float],
    batch_size_per_gpu: int,
    num_classes: int,
    device: torch.device,
    use_avgpool: bool,
) -> Tuple[nn.Module, List[Dict]]:
    classifiers_dict = {}
    optim_param_groups = []

    for n_blocks in n_last_blocks_list:
        for base_lr in learning_rates:
            lr = scale_lr(base_lr, batch_size_per_gpu)
            out_dim = create_linear_input(sample_output, n_blocks, use_avgpool).shape[1]
            classifier = LinearClassifier(
                out_dim=out_dim,
                use_n_blocks=n_blocks,
                use_avgpool=use_avgpool,
                num_classes=num_classes,
            ).to(device)
            key = f"classifier_{n_blocks}_blocks_avgpool_{use_avgpool}_lr_{lr:.6g}".replace(".", "_")
            classifiers_dict[key] = classifier
            optim_param_groups.append({"params": classifier.parameters(), "lr": lr})
            if is_main_process():
                LOGGER.info("Create %s with input_dim=%d", key, out_dim)

    linear_classifiers = AllClassifiers(classifiers_dict)
    if is_dist_avail_and_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        linear_classifiers = nn.parallel.DistributedDataParallel(
            linear_classifiers, device_ids=[local_rank]
        )

    return linear_classifiers, optim_param_groups


def train_one_epoch(
    feature_model: nn.Module,
    linear_classifiers: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    train_loader: DataLoader,
    epoch: int,
    epoch_length: int,
    device: torch.device,
) -> float:
    linear_classifiers.train()
    total_loss = 0.0
    num_batches = 0

    iterator = enumerate(train_loader)
    if is_main_process():
        iterator = enumerate(tqdm(train_loader, total=epoch_length, desc=f"Epoch {epoch}"))

    for batch_idx, (images, labels) in iterator:
        if batch_idx >= epoch_length:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        features = feature_model(images)
        outputs = linear_classifiers(features)
        loss = sum(criterion(logits, labels) for logits in outputs.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.detach().item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    feature_model: nn.Module,
    linear_classifiers: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    linear_classifiers.eval()
    classifiers_dict = (
        linear_classifiers.module.classifiers_dict
        if hasattr(linear_classifiers, "module")
        else linear_classifiers.classifiers_dict
    )

    correct = {k: 0 for k in classifiers_dict.keys()}
    total = 0

    iterator = val_loader
    if is_main_process():
        iterator = tqdm(val_loader, desc="Evaluating")

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        features = feature_model(images)
        outputs = linear_classifiers(features)

        for key, logits in outputs.items():
            pred = logits.argmax(dim=1)
            correct[key] += int((pred == labels).sum().item())
        total += int(labels.numel())

    if is_dist_avail_and_initialized():
        total_tensor = torch.tensor([total], device=device, dtype=torch.long)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        total = int(total_tensor.item())

        for key in correct:
            correct_tensor = torch.tensor([correct[key]], device=device, dtype=torch.long)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            correct[key] = int(correct_tensor.item())

    return {key: 100.0 * value / max(total, 1) for key, value in correct.items()}


def run_linear_probe(args) -> Dict[str, object]:
    device = init_device(args.use_ddp, args.device)
    setup_logging()
    seed_everything(args.seed)
    cudnn.benchmark = True

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    if is_main_process():
        LOGGER.info("Loading VTP model from %s", args.model_path)
    model = VTPModel.from_pretrained(args.model_path).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    autocast_dtype = precision_to_dtype(args.precision)
    feature_model = FeatureExtractor(model, max(args.n_last_blocks_list), autocast_dtype).to(device)

    train_path = os.path.join(args.imagenet_root, "train")
    val_path = os.path.join(args.imagenet_root, "val")
    train_dataset = ImageFolder(root=train_path, transform=make_train_transform(args.crop_size))
    val_dataset = ImageFolder(root=val_path, transform=make_eval_transform(args.resize_size, args.crop_size))
    num_classes = len(train_dataset.classes)

    if val_dataset.class_to_idx != train_dataset.class_to_idx:
        raise RuntimeError("ImageNet train/val class_to_idx mismatch. Use ImageFolder-style train/val folders.")

    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=InfiniteSampler(train_sampler),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    sample_image = train_dataset[0][0].unsqueeze(0).to(device)
    sample_output = feature_model(sample_image)
    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output=sample_output,
        n_last_blocks_list=args.n_last_blocks_list,
        learning_rates=args.learning_rates,
        batch_size_per_gpu=args.batch_size,
        num_classes=num_classes,
        device=device,
        use_avgpool=args.avgpool_patchtokens,
    )

    max_iter = args.epochs * args.epoch_length
    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0.0)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    best_classifier = ""
    last_accuracies = {}

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            feature_model=feature_model,
            linear_classifiers=linear_classifiers,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_loader=train_loader,
            epoch=epoch,
            epoch_length=args.epoch_length,
            device=device,
        )
        accuracies = evaluate(feature_model, linear_classifiers, val_loader, device)
        last_accuracies = accuracies
        current_best_key = max(accuracies, key=accuracies.get)
        current_best_acc = accuracies[current_best_key]
        if current_best_acc > best_accuracy:
            best_accuracy = current_best_acc
            best_classifier = current_best_key

        if is_main_process():
            log_stats = {
                "epoch": epoch,
                "train_loss": train_loss,
                "current_best_accuracy": current_best_acc,
                "current_best_classifier": current_best_key,
                "best_accuracy": best_accuracy,
                "best_classifier": best_classifier,
            }
            print(json.dumps(log_stats, indent=2))
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    results = {
        "best_accuracy": best_accuracy,
        "best_classifier": best_classifier,
        "all_accuracies": last_accuracies,
        "feature_protocol": "VTP/DINO intermediate cls tokens + optional patch-token avgpool",
        "avgpool_patchtokens": args.avgpool_patchtokens,
        "n_last_blocks_list": list(args.n_last_blocks_list),
        "learning_rates": list(args.learning_rates),
    }

    if is_main_process():
        with open(os.path.join(args.output_dir, "linear_probing_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print("========== Final ==========")
        print(json.dumps(results, indent=2))

    if args.use_ddp and is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return results


def parse_args():
    parser = argparse.ArgumentParser("Original VTP/DINO-style linear probing with avgpool patch tokens")
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--imagenet-root", "--imagenet_root", dest="imagenet_root", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--epoch-length", "--epoch_length", dest="epoch_length", type=int, default=1250)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--precision", default="bf16", choices=["fp32", "float32", "fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--crop-size", dest="crop_size", type=int, default=CROP_SIZE)
    parser.add_argument("--resize-size", dest="resize_size", type=int, default=RESIZE_SIZE)

    parser.add_argument("--n-last-blocks-list", "--n_last_blocks_list", dest="n_last_blocks_list", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--learning-rates", "--learning_rates", dest="learning_rates", type=float, nargs="+", default=list(DEFAULT_LEARNING_RATES))
    parser.add_argument("--avgpool-patchtokens", dest="avgpool_patchtokens", action="store_true", default=True)
    parser.add_argument("--no-avgpool-patchtokens", dest="avgpool_patchtokens", action="store_false")
    parser.add_argument("--use-ddp", "--use_ddp", dest="use_ddp", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run_linear_probe(parse_args())
