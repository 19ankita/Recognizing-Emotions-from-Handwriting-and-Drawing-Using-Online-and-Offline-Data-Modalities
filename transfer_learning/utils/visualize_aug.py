import os
import cv2
import yaml
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
# Build SAME augmentations used in get_transforms()
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
            A.ElasticTransform(alpha=10, sigma=50, alpha_affine=10, p=1.0),
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
            A.CoarseDropout(
                max_holes=2, 
                max_height=img_size//8, 
                max_width=img_size//8, 
                p=1.0
            ),
            A.Resize(img_size, img_size), A.Normalize(mean,std), ToTensorV2()
        ]),

        # Full Pipeline — EXACT match to your training pipeline
        "Full Pipeline": A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=8, p=0.8),
            A.ElasticTransform(alpha=10, sigma=50, alpha_affine=10, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.CoarseDropout(max_holes=2, max_height=img_size//8, max_width=img_size//8, p=0.5),

            A.Resize(img_size, img_size),
            A.Normalize(mean,std),
            ToTensorV2()
        ])
    }


# -----------------------------------------------------
# MAIN VISUALIZATION FUNCTION
# -----------------------------------------------------
def visualize_augmentations(config_path):

    cfg = yaml.safe_load(open(config_path, "r"))
    img_size = cfg["img_size"]

    # Load any sample image
    root = cfg["task_dir"]
    cls = os.listdir(root)[0]
    img_path = os.path.join(root, cls, os.listdir(os.path.join(root, cls))[0])

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    augs = build_train_augs(img_size)

    plt.figure(figsize=(20, 12))

    for idx, (name, aug) in enumerate(augs.items(), 1):
        out = aug(image=img)["image"]   # tensor (C,H,W)

        # Convert to HWC
        if out.dim() == 3:
            out_img = denormalize(out)
        else:
            out_img = out

        plt.subplot(3, 3, idx)
        plt.imshow(out_img)
        plt.title(name, fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    output_path = os.path.join("outputs", "augmentations.pdf")
    plt.savefig(output_path, dpi=300, format="pdf")
    print(f"Saved augmentation visualization → {output_path}")

    plt.show()


# -----------------------------------------------------
# Entry Point
# -----------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    visualize_augmentations(args.config)
