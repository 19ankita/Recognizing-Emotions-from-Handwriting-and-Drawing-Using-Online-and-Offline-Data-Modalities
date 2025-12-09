import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import yaml
import os
import numpy as np

from src.dataset import Cutout  # reuse your Cutout class


def denormalize(tensor):
    """Convert normalized tensor back to image format."""
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
    x = tensor * std + mean
    return x.permute(1,2,0).clamp(0,1)


def visualize_augmentations(config_path):
    cfg = yaml.safe_load(open(config_path, "r"))
    img_size = cfg["img_size"]

    # Load sample image (first class, first image)
    root = cfg["task_dir"]
    class_name = os.listdir(root)[0]
    class_dir = os.path.join(root, class_name)
    img_path = os.path.join(class_dir, os.listdir(class_dir)[0])

    img = Image.open(img_path).convert("RGB")

    # ---------------------------
    # DEFINE EVERY AUGMENTATION
    # ---------------------------
    aug_original = transforms.ToTensor()

    aug_crop = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.ToTensor()
    ])

    aug_hflip = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor()
    ])

    aug_rotate = transforms.Compose([
        transforms.RandomRotation(8),
        transforms.ToTensor()
    ])

    aug_affine = transforms.Compose([
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            shear=5
        ),
        transforms.ToTensor()
    ])

    aug_jitter = transforms.Compose([
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor()
    ])

    aug_cutout = transforms.Compose([
        transforms.ToTensor(),
        Cutout(n_holes=1, length=25)
    ])

    aug_full_pipeline = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(8),
        transforms.RandomAffine(degrees=0, translate=(0.08,0.08), shear=5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=25),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    # List of labeled transforms
    transforms_list = [
        ("Original", aug_original),
        ("RandomResizedCrop", aug_crop),
        ("HorizontalFlip", aug_hflip),
        ("Rotation", aug_rotate),
        ("Affine (Translate + Shear)", aug_affine),
        ("ColorJitter", aug_jitter),
        ("Cutout", aug_cutout),
        ("Full Pipeline", aug_full_pipeline)
    ]

    # ---------------------------
    # VISUALIZE
    # ---------------------------
    plt.figure(figsize=(18, 10))
    for i, (label, tf) in enumerate(transforms_list, 1):
        tensor = tf(img)
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            if "Full Pipeline" in label:
                tensor = denormalize(tensor)
            else:
                tensor = tensor.permute(1, 2, 0).clamp(0, 1)

        plt.subplot(2, 4, i)
        plt.imshow(tensor)
        plt.title(label, fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    visualize_augmentations(args.config)
