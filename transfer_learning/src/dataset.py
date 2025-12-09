import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np

from torchvision.transforms.autoaugment import RandAugment


# ---------------------------------------------------------
# Cutout Augmentation (must be applied AFTER ToTensor)
# ---------------------------------------------------------
class Cutout(object):
    def __init__(self, n_holes=1, length=30):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        # img is a tensor: C x H x W
        _, h, w = img.shape

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask).float()
        mask = mask.expand_as(img)

        return img * mask


# ---------------------------------------------------------
# Build Train & Validation Transform Pipelines
# ---------------------------------------------------------
def get_transforms(cfg):
    img_size = cfg["img_size"]

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # -------------------------------
    # STRONG TRAIN AUGMENTATION
    # -------------------------------
    # train_tf = transforms.Compose([
    #     transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
    #     transforms.RandomHorizontalFlip(p=0.5),

    #     # Powerful augmentation from Google Brain
    #     RandAugment(num_ops=2, magnitude=9),

    #     transforms.RandomAffine(
    #         degrees=10,
    #         translate=(0.08, 0.08),
    #         shear=8
    #     ),

    #     transforms.ColorJitter(
    #         brightness=0.2,
    #         contrast=0.2,
    #         saturation=0.2
    #     ),

    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std),

    #     # Additional regularization
    #     Cutout(n_holes=1, length=20)
    # ])
    
    train_tf = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        shear=10
    ),
    transforms.ColorJitter(contrast=0.3, brightness=0.2),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=24),
    transforms.Normalize(mean, std)
])


    # -------------------------------
    # Validation transforms (clean)
    # -------------------------------
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_tf, val_tf


# ---------------------------------------------------------
# Create DataLoaders
# ---------------------------------------------------------
def get_dataloaders(cfg):
    train_tf, val_tf = get_transforms(cfg)

    # Base dataset without transforms
    full_dataset = ImageFolder(cfg["task_dir"], transform=None)

    # Train/Val split
    val_ratio = cfg.get("val_ratio", 0.2)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Assign transforms to split datasets
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform = val_tf

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"]
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"]
    )

    num_classes = len(full_dataset.classes)

    return train_loader, val_loader, num_classes
