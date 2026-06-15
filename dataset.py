"""
dataset.py — GazeDataset, train/val/test split, and DataLoader factory.

Assumes all_paired_data = [(img_path, ann_path), ...]
where ann_path is a *_center.txt file containing:
    "x, y"   (already normalized to [0, 1])

Key improvements over v1:
  - PadToSquare: preserves aspect ratio before resize (40x20 → 40x40 → 224x224
    instead of 40x20 → 224x224; halves the upscale artefacts on y-axis)
  - BICUBIC interpolation: sharper edges after large upscaling
  - Paired horizontal flip: flips the image AND mirrors the x label (1-x),
    doubling effective dataset size without corrupting labels
  - Stronger colour/blur augmentation: reduces overfitting on tiny images
  - RandomErasing: simulates eyelid / reflection occlusions
"""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_INPUT_SIZE    = 224


# ---------------------------------------------------------------------------
# PadToSquare — aspect-ratio-preserving resize helper
# ---------------------------------------------------------------------------

class PadToSquare:
    """
    Pad the shorter side with black pixels so the image becomes square,
    then the standard Resize can upscale uniformly on both axes.

    Example: 40x20 → pad top+bottom by 10px → 40x40 → resize 224x224
             Upscale factor: 5.6x (was 11x on height without this step).
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        pad_l = (size - w) // 2
        pad_t = (size - h) // 2
        new_img = Image.new(img.mode, (size, size), 0)
        new_img.paste(img, (pad_l, pad_t))
        return new_img


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GazeDataset(Dataset):
    """
    Loads (eye image, pupil center label) pairs.

    Args:
        paired_data : list of (img_path, ann_path) tuples
        transform   : torchvision transform applied AFTER optional flip
        flip_aug    : if True, randomly mirror image + correct x label (train only)
    """

    def __init__(self, paired_data: list, transform=None, flip_aug: bool = False):
        self.paired_data = paired_data
        self.transform   = transform if transform is not None else _eval_transform()
        self.flip_aug    = flip_aug

    def __len__(self) -> int:
        return len(self.paired_data)

    def __getitem__(self, idx: int):
        img_path, ann_path = self.paired_data[idx]

        img = Image.open(img_path).convert("RGB")

        # Read label BEFORE any spatial transform that could affect it
        raw = Path(ann_path).read_text().strip().split(",")
        x, y = float(raw[0]), float(raw[1])

        # Paired horizontal flip: mirror image ↔ mirror x label
        # Both left-eye and right-eye images are in the dataset, so flipping
        # one type makes it look like the other — a valid data augmentation.
        if self.flip_aug and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            x = 1.0 - x

        img   = self.transform(img)
        label = torch.tensor([x, y], dtype=torch.float32)
        return img, label


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def _train_transform():
    """
    Augmentation pipeline for training. All non-spatial transforms that
    do NOT require label adjustments (colour/blur/erase only).
    PadToSquare + flip handle the spatial side, and flip is done in __getitem__.
    """
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.08),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        # Simulate eyelid/glare occlusion; applied in tensor space
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10),
                                 ratio=(0.3, 3.3), value=0),
    ])


def _eval_transform():
    """Deterministic transform for validation and test sets."""
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def split_data(all_paired_data: list,
               train_ratio: float = 0.70,
               val_ratio: float = 0.15,
               seed: int = 42) -> tuple[list, list, list]:
    """Shuffle + split all_paired_data into (train, val, test) lists."""
    data = list(all_paired_data)
    random.seed(seed)
    random.shuffle(data)

    n       = len(data)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_data = data[:n_train]
    val_data   = data[n_train : n_train + n_val]
    test_data  = data[n_train + n_val :]

    print(f"[Dataset] Split — train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(all_paired_data: list,
                    batch_size: int = 8,
                    num_workers: int = 0,
                    train_ratio: float = 0.70,
                    val_ratio: float = 0.15,
                    seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    batch_size=8 is the safe default for CPU (EfficientNet-B0 backward
    peaks at ~700 MB; batch_size=32 OOM-kills the kernel on most machines).
    """
    train_data, val_data, test_data = split_data(
        all_paired_data, train_ratio, val_ratio, seed
    )

    train_ds = GazeDataset(train_data, transform=_train_transform(), flip_aug=True)
    val_ds   = GazeDataset(val_data,   transform=_eval_transform(),  flip_aug=False)
    test_ds  = GazeDataset(test_data,  transform=_eval_transform(),  flip_aug=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    return train_loader, val_loader, test_loader
