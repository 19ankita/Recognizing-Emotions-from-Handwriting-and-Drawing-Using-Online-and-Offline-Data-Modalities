import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch


def get_transforms(cfg):

    img_size = cfg["img_size"]

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # STRONG AUGMENTATION FOR TRAINING
    train_tf = A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.10, rotate_limit=8, p=0.8
        ),
        A.ElasticTransform(
            alpha=10, sigma=50, alpha_affine=10, p=0.5
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
        A.CoarseDropout(max_holes=2, max_height=img_size//8, max_width=img_size//8, p=0.5),
        A.Resize(img_size, img_size),
        A.Normalize(mean, std),
        ToTensorV2()
    ])

    # Validation: mild geometric normalization only
    val_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean, std),
        ToTensorV2()
    ])

    return train_tf, val_tf


class AlbumentationsDataset(ImageFolder):
    """Apply Albumentations instead of torchvision transforms"""

    def __init__(self, root, transform=None):
        super().__init__(root)
        self.albumentations_transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.albumentations_transform:
            image = self.albumentations_transform(image=image)["image"]

        return image, label



def get_dataloaders(cfg):

    train_tf, val_tf = get_transforms(cfg)

    full_dataset = AlbumentationsDataset(cfg["task_dir"], transform=None)

    val_size = int(len(full_dataset) * cfg["val_ratio"])
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Attach Albumentations transforms
    train_ds.dataset.albumentations_transform = train_tf
    val_ds.dataset.albumentations_transform   = val_tf

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg["num_workers"])

    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"])

    return train_loader, val_loader, len(full_dataset.classes)
