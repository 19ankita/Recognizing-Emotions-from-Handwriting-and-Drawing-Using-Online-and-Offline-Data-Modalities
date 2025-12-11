import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


# -----------------------------------------------------
# De-normalization helper
# -----------------------------------------------------
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
    x = tensor * std + mean
    return x.permute(1, 2, 0).clamp(0, 1).cpu().numpy()


# -----------------------------------------------------
# Build SAME augmentations used in dataset.py
# -----------------------------------------------------
def build_train_augs(img_size):

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    return {
        "ShiftScaleRotate": A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=8, p=1.0),
            A.Resize(img_size, img_size), A.Normalize(mean,std), ToTensorV2()
        ]),

        "ElasticTransform": A.Compose([
            A.ElasticTransform(alpha=10, sigma=50, p=1.0),
            A.Resize(img_size, img_size), A.Normalize(mean,std), ToTensorV2()
        ]),

        "GridDistortion": A.Compose([
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
            A.Resize(img_size, img_size), A.Normalize(mean,std), ToTensorV2()
        ]),

        "Perspective": A.Compose([
            A.Perspective(scale=(0.02, 0.05), p=1.0),
            A.Resize(img_size, img_size), A.Normalize(mean,std), ToTensorV2()
        ]),

        "BrightnessContrast": A.Compose([
            A.RandomBrightnessContrast(0.1, 0.1, p=1.0),
            A.Resize(img_size, img_size), A.Normalize(mean,std), ToTensorV2()
        ]),

        "CoarseDropout": A.Compose([
            A.CoarseDropout(max_holes=2, max_height=img_size//8, max_width=img_size//8, p=1.0),
            A.Resize(img_size, img_size), A.Normalize(mean,std), ToTensorV2()
        ]),

        # Full training pipeline
        "Full Pipeline": A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=8, p=0.8),
            A.ElasticTransform(alpha=10, sigma=50, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.CoarseDropout(max_holes=2, max_height=img_size//8, max_width=img_size//8, p=0.5),

            A.Resize(img_size, img_size),
            A.Normalize(mean,std),
            ToTensorV2()
        ]),
    }


# -----------------------------------------------------
# AUGMENTATION VISUALIZATION (NO YAML)
# -----------------------------------------------------
def visualize_augmentations(task, task_dir, img_size):

    # -------------------------------------------------
    # Load a sample image
    # -------------------------------------------------
    if task == "all":
        subfolders = sorted([
            f for f in os.listdir(task_dir)
            if os.path.isdir(os.path.join(task_dir, f))
        ])
        first_task = subfolders[0]
        folder = os.path.join(task_dir, first_task)
    else:
        folder = os.path.join(task_dir, task)

    class_names = os.listdir(folder)
    first_class = class_names[0]

    img_path = os.path.join(folder, first_class, os.listdir(os.path.join(folder, first_class))[0])

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    augs = build_train_augs(img_size)

    plt.figure(figsize=(20, 12))

    # ORIGINAL
    plt.subplot(3, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    # AUGMENTED IMAGES
    for idx, (name, aug) in enumerate(augs.items(), start=2):
        out_tensor = aug(image=img)["image"]
        out_img = denormalize(out_tensor)

        plt.subplot(3, 3, idx)
        plt.imshow(out_img)
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    output_path = os.path.join("outputs", "augmentations.pdf")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved augmentation visualization → {output_path}")
    plt.show()


# -----------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--task_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)

    args = parser.parse_args()

    visualize_augmentations(args.task, args.task_dir, args.img_size)
