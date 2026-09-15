import io
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import PIL.Image
import random
from glob import glob
from PIL import Image
from torchvision.datasets import VisionDataset
import torchvision.transforms.v2 as transforms
from pathlib import Path
from typing import List, Optional, Union, Tuple
import webdataset as wds
from torchvision import transforms as transformsv1
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
from torch.utils.data.distributed import DistributedSampler

from pathlib import Path
from typing import Callable, Optional, Union

import torch
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision import transforms


# Dataset metadata for size estimation
# Note: num_shards are actual counts; num_samples are estimates (used for epoch length calculation)
BLIP3O_METADATA = {
    # "journeydb": {"num_samples": 4_280_000, "num_shards": 419},
    # "short-caption": {"num_samples": 4_770_000, "num_shards": 1831},
    # "long-caption": {"num_samples": 27_200_000, "num_shards": 2891},
    # "60k": {"num_samples": 57_553, "num_shards": 11},
    # "60k-128shards": {"num_samples": 57_553, "num_shards": 128},  # Resharded for better worker utilization
}

class ImageNetValDataset(Dataset):
    def __init__(self, root, transform=None, small=-1):
        """
        Args:
            root (str): Path to ImageNet val directory (e.g. /data/imagenet/val)
            transform (callable, optional): Transform to apply to images
        """
        self.root = root
        self.transform = transform
        self.small = small

        self.samples = []
        self.class_to_idx = {}

        classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_dir = os.path.join(root, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(cls_dir, fname)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))

        if self.small > 0:
            random.Random(42).shuffle(self.samples)

    def __len__(self):
        if self.small > 0:
            return self.small
        else:
            return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = PIL.Image.open(path).convert("RGB")
        name = str(path).split("/")[-1]

        if self.transform:
            image = self.transform(image)

        return image, label, name


def load_h5_file(hf, path):
    # Helper function to load files from h5 file
    if path.endswith(".png"):
        rtn = np.array(PIL.Image.open(io.BytesIO(np.array(hf[path]))))
        rtn = rtn.reshape(*rtn.shape[:2], -1).transpose(2, 0, 1)
    elif path.endswith(".json"):
        rtn = json.loads(np.array(hf[path]).tobytes().decode("utf-8"))
    elif path.endswith(".npy"):
        rtn = np.array(hf[path])
    else:
        raise ValueError("Unknown file type: {}".format(path))
    return rtn


class CustomINH5Dataset(Dataset):
    def __init__(self, data_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {".npy"}

        self.data_dir = data_dir
        self.h5_path = os.path.join(self.data_dir, "images.h5")
        self.h5_json_path = os.path.join(self.data_dir, "images_h5.json")
        self.h5f = h5py.File(self.h5_path, "r")

        with open(self.h5_json_path, "r") as f:
            self.h5_json = json.load(f)
        self.filelist = {fname for fname in self.h5_json}
        self.filelist = sorted(
            fname for fname in self.filelist if self._file_ext(fname) in supported_ext
        )

        labels = load_h5_file(self.h5f, "dataset.json")["labels"]
        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self.filelist]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def __len__(self):
        return len(self.filelist)

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __del__(self):
        self.h5f.close()

    def __getitem__(self, index):
        """
        Images should be '.png'
        """
        image_fname = self.filelist[index]
        image = load_h5_file(self.h5f, image_fname)
        return torch.from_numpy(image), torch.tensor(self.labels[index])


class CustomH5Dataset(Dataset):
    def __init__(self, data_dir, vae_latents_name="repae-invae-400k"):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {".npy"}

        self.images_h5 = h5py.File(os.path.join(data_dir, "images.h5"), "r")
        self.features_h5 = h5py.File(
            os.path.join(data_dir, f"{vae_latents_name}.h5"), "r"
        )
        images_json = os.path.join(data_dir, "images_h5.json")
        features_json = os.path.join(data_dir, f"{vae_latents_name}_h5.json")

        with open(images_json, "r") as f:
            images_json = json.load(f)
        with open(features_json, "r") as f:
            features_json = json.load(f)

        # images
        self._image_fnames = {fname for fname in images_json}
        self.image_fnames = sorted(
            fname
            for fname in self._image_fnames
            if self._file_ext(fname) in supported_ext
        )

        # features
        self._feature_fnames = {fname for fname in features_json}
        self.feature_fnames = sorted(
            fname
            for fname in self._feature_fnames
            if self._file_ext(fname) in supported_ext
        )

        # labels
        fname = "dataset.json"
        labels = load_h5_file(self.features_h5, fname)["labels"]
        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), (
            "Number of feature files and label files should be same"
        )
        return len(self.feature_fnames)

    def __del__(self):
        self.images_h5.close()
        self.features_h5.close()

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)

        image = load_h5_file(self.images_h5, image_fname)
        if image_ext == ".npy":
            # npy needs some extra care
            image = image.reshape(-1, *image.shape[-2:])

        features = load_h5_file(self.features_h5, feature_fname)
        return (
            torch.from_numpy(image),
            torch.from_numpy(features),
            torch.tensor(self.labels[idx]),
        )

class SimpleDataset(VisionDataset):
    def __init__(self, root: str, image_size):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        if root.endswith(".txt"):
            with open(root) as f:
                lines = f.readlines()
            self.fpaths = [line.strip("\n") for line in lines]
        else:
            self.fpaths = sorted(glob(root + "/**/*.JPEG", recursive=True))
            self.fpaths += sorted(glob(root + "/**/*.jpg", recursive=True))
            self.fpaths += sorted(glob(root + "/**/*.png", recursive=True))

        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        try:
            fpath = self.fpaths[index]
            img = Image.open(fpath).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading image {fpath}: {e}")
            return self.__getitem__((index + 1) % len(self.fpaths))

        return img



def _filter_valid_samples(sample):
    """Filter function for .select() - must be module-level to be pickleable."""
    return sample[0] is not None

class BLIP3OWebDataset:
    """
    WebDataset wrapper for BLIP3O splits.
    Supports combining multiple splits and uses wds.split_by_node for DDP.

    Returns (image_tensor, caption_string) pairs compatible with existing
    text conditioning interface.
    """

    def __init__(
        self,
        data_dir: str,
        splits: Union[str, List[str]],
        transform: Optional[transformsv1.Compose] = None,
        image_size: int = 256,
        shuffle_buffer: int = 20000,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Base path to BLIP3O data (e.g., 'data/blip3o')
            splits: Single split name or list of splits to combine
                   Options: 'journeydb', 'short-caption', 'long-caption', '60k'
            transform: Optional custom transform. If None, uses default augmentation.
            image_size: Target image resolution
            shuffle_buffer: Size of shuffle buffer for sample-level shuffling
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.splits = [splits] if isinstance(splits, str) else list(splits)
        self.transform = transform
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        # Collect shard URLs and calculate total samples
        self._total_samples = 0
        self._shard_urls = []

        for split in self.splits:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                raise ValueError(f"Split directory not found: {split_dir}")

            tar_files = sorted(split_dir.glob("*.tar"))
            self._shard_urls.extend([str(f) for f in tar_files])

            if split in BLIP3O_METADATA:
                self._total_samples += BLIP3O_METADATA[split]["num_samples"]
            else:
                # Fallback estimate: ~3500 samples per shard (BLIP3O average)
                self._total_samples += len(tar_files) * 3500

        if not self._shard_urls:
            raise ValueError(f"No tar shards found for splits {self.splits} in {data_dir}")

        self._num_shards = len(self._shard_urls)

        # Default transform with augmentation
        if self.transform is None:
            first_crop = int(image_size * 1.5)
            self.transform = transformsv1.Compose([
                transformsv1.Resize(first_crop, interpolation=transformsv1.InterpolationMode.BICUBIC),
                transformsv1.RandomCrop(image_size),
                transformsv1.RandomHorizontalFlip(),
                transformsv1.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ])

    @property
    def estimated_size(self) -> int:
        """Return estimated total samples (for epoch calculation)."""
        return self._total_samples

    @property
    def num_shards(self) -> int:
        """Return number of shards."""
        return self._num_shards

    def _decode_sample(self, sample):
        """Decode WebDataset sample (image + txt pair)."""
        # Get image - handle different extensions
        image = sample.get("jpg") or sample.get("png") or sample.get("jpeg") or sample.get("webp")

        # Get caption - handle bytes or string
        caption_raw = sample.get("txt", b"")
        if isinstance(caption_raw, bytes):
            caption = caption_raw.decode("utf-8").strip()
        else:
            caption = str(caption_raw).strip()

        # Apply transform if image exists
        if self.transform is not None and image is not None:
            image = self.transform(image)

        return image, caption

    def create_pipeline(self, epoch: int = 0) -> wds.WebDataset:
        """
        Create WebDataset pipeline for a given epoch.

        Call at start of each epoch to ensure proper shuffling with deterministic seed.
        """
        dataset = (
            wds.WebDataset(
                self._shard_urls,
                nodesplitter=wds.split_by_node,
                shardshuffle=1000,  # Shuffle buffer size for shards
                seed=self.seed + epoch,
            )
            .shuffle(self.shuffle_buffer, initial=self.shuffle_buffer // 2)
            .decode("pil", handler=wds.ignore_and_continue)  # Skip corrupt images
            .map(self._decode_sample, handler=wds.ignore_and_continue)
            # .select(lambda x: x[0] is not None)  # Filter failed decodes
            .select(_filter_valid_samples)  # Filter failed decodes (lambda function is not pickleable for spawn)
        )
        return dataset
    
class _T2IHFDataset(Dataset):
    """Internal HuggingFace dataset wrapper for MSCOCO/MJHQ T2I datasets."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "val",
        transform: Optional[transformsv1.Compose] = None,
        data_dir: Optional[str] = "./data",
    ):
        from datasets import load_from_disk

        # Load from local Arrow format (e.g., data/mscoco/val)
        local_path = Path(data_dir) / dataset_name / split
        self.hf_dataset = load_from_disk(str(local_path))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        sample = self.hf_dataset[idx]
        text = sample['text']

        if 'image' in sample:
            image = sample['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        else:
            image = torch.empty(0)

        return image, text

@dataclass
class DataloaderResult:
    """
    Unified result from prepare_unified_dataloader.
    Provides consistent interface for map-style and iterable datasets.
    """
    loader: Union[DataLoader, wds.WebLoader]
    sampler: Optional[DistributedSampler]
    dataset_size: int
    is_iterable: bool = False
    _wds_pipeline: Optional[object] = field(default=None, repr=False)
    _batch_size: int = 1
    _num_workers: int = 4
    _world_size: int = 1
    virtual_epoch_steps: Optional[int] = None

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling. Works for both dataset types.

        For map-style: calls sampler.set_epoch()
        For WebDataset: recreates pipeline with new seed (uses virtual_epoch_steps if set)
        """
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        elif self._wds_pipeline is not None:
            self._recreate_wds_loader(epoch)

    def _recreate_wds_loader(self, epoch: int):
        """Recreate WebDataset loader for new epoch."""
        dataset = self._wds_pipeline.create_pipeline(epoch=epoch)
        steps = self.virtual_epoch_steps or (self.dataset_size // (self._batch_size * self._world_size))
        loader = wds.WebLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        self.loader = loader.with_epoch(steps)

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        if self.virtual_epoch_steps is not None:
            return self.virtual_epoch_steps
        if self.is_iterable:
            return self.dataset_size // (self._batch_size * self._world_size)
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

def _prepare_blip3o_loader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    world_size: int,
    transform: Optional[transforms.Compose],
) -> DataloaderResult:
    """Prepare BLIP3O WebDataset loader."""

    data_dir = config.get("data_dir", "./data/blip3o")
    # Support both 'splits' (list) and 'split' (single) keys
    splits = config.get("splits", config.get("split", "short-caption"))
    shuffle_buffer = config.get("shuffle_buffer", 10000)
    seed = config.get("seed", 42)

    wds_pipeline = BLIP3OWebDataset(
        data_dir=data_dir,
        splits=splits,
        transform=transform,
        image_size=image_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    dataset = wds_pipeline.create_pipeline(epoch=0)
    total_samples = wds_pipeline.estimated_size
    steps = total_samples // (batch_size * world_size)

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    # Bound epoch to exactly `steps` batches (with_epoch stops iteration, with_length only sets __len__)
    loader = loader.with_epoch(steps)

    return DataloaderResult(
        loader=loader,
        sampler=None,
        dataset_size=total_samples,
        is_iterable=True,
        _wds_pipeline=wds_pipeline,
        _batch_size=batch_size,
        _num_workers=num_workers,
        _world_size=world_size,
    )


def _prepare_t2i_hf_loader(
    dataset_name: str,
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    transform: Optional[transformsv1.Compose],
    shuffle: bool,
) -> DataloaderResult:
    """Prepare MSCOCO/MJHQ HuggingFace loader."""
    split = config.get("split", "val")
    data_dir = config.get("data_dir", "./data")  # Local data directory

    if transform is None:
        transform = transformsv1.Compose([
            transformsv1.Resize(image_size, interpolation=transformsv1.InterpolationMode.BICUBIC),
            transformsv1.CenterCrop(image_size),
            transformsv1.ToTensor(),
        ])

    dataset = _T2IHFDataset(
        dataset_name=dataset_name,
        split=split,
        transform=transform,
        data_dir=data_dir,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # drop_last=True for train, False for eval
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    return DataloaderResult(
        loader=loader,
        sampler=sampler,
        dataset_size=len(dataset),
        is_iterable=False,
    )

if __name__ == "__main__":
    dataset_config = {
        "data_dir": "/mnt/task_runtime",
        "splits": ["BLIP3o-Pretrain-JourneyDB", "BLIP3o-Pretrain-Long-Caption", "BLIP3o-Pretrain-Short-Caption"]
    }
    blip3dataset = _prepare_blip3o_loader(
        dataset_config,
        image_size=256, 
        batch_size=256,
        num_workers=16,
        world_size=8,
        transform=None,
    )
    print(len(blip3dataset))



class FFHQ256Dataset(Dataset):
    """
    PyTorch dataset for the locally downloaded Hugging Face dataset:
        bitmind/ffhq-256

    Expected directory:
        /mnt/task_runtime/datasets/ffhq/
            data/
                train-00000-of-00016.parquet
                ...
            README.md
    """

    def __init__(
        self,
        root: Union[str, Path] = "/mnt/task_runtime/datasets/ffhq",
        transform: Optional[Callable] = None,
        image_column: str = "image",
        return_dict: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_column = image_column
        self.return_dict = return_dict

        parquet_files = sorted((self.root).glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in {self.root}"
            )

        # Load all local parquet shards as one map-style Hugging Face Dataset.
        self.dataset = load_dataset(
            "parquet",
            data_files={"train": [str(path) for path in parquet_files]},
            split="train",
        )

        if self.image_column not in self.dataset.column_names:
            raise KeyError(
                f"Image column {self.image_column!r} was not found. "
                f"Available columns: {self.dataset.column_names}"
            )

        # Ensure the image column is decoded as a PIL image.
        if not isinstance(self.dataset.features[self.image_column], HFImage):
            self.dataset = self.dataset.cast_column(
                self.image_column,
                HFImage(),
            )

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(
                    (256, 256),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        example = self.dataset[index]
        image = example[self.image_column]

        if not isinstance(image, PILImage.Image):
            raise TypeError(
                f"Expected a PIL image, but got {type(image)} "
                f"from column {self.image_column!r}"
            )

        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, 0