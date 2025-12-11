import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.datasets import ImageFolder
import torch
import os


def get_transforms(img_size):

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
        
        A.CoarseDropout(max_holes=2, 
                        max_height=img_size//8,
                        max_width=img_size//8, 
                        p=0.5),
        
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


# ------------------------------------------------------------
# Albumentations wrapper for ImageFolder
# ------------------------------------------------------------
class AlbumentationsDataset(ImageFolder):

    def __init__(self, root, transform=None):
        super().__init__(root)
        self.albu_transform = transform

    def set_transform(self, tf):
        self.albu_transform = tf

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.albu_transform:
            image = self.albu_transform(image=image)["image"]

        return image, label



def get_dataloaders(task, task_root, img_size, batch_size, num_workers, val_ratio):

    train_tf, val_tf = get_transforms(img_size)

    # ------------------------------
    # CASE 1: SINGLE TASK
    # ------------------------------
    if task != "all":
        task_dir = os.path.join(task_root, task)
        print(f"Loading single task: {task_dir}")
        dataset = AlbumentationsDataset(task_dir)

    # ------------------------------
    # CASE 2: ALL TASKS COMBINED
    # ------------------------------
    else:
        print("Loading ALL tasks...")
        subfolders = sorted([
            f for f in os.listdir(task_root)
            if f.startswith("task") and os.path.isdir(os.path.join(task_root, f))
        ])

        datasets = []
        for sub in subfolders:
            path = os.path.join(task_root, sub)
            print(" :", path)
            datasets.append(AlbumentationsDataset(path))

        dataset = ConcatDataset(datasets)

    # ------------------------------
    # TRAIN/VAL SPLIT
    # ------------------------------
    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    # Case 1: ConcatDataset
    if isinstance(dataset, ConcatDataset):
        # All sub-datasets inside ConcatDataset get TRAIN transform
        for ds in dataset.datasets:
            ds.set_transform(train_tf)

        # Validation dataset must use val_tf
        val_ds.dataset.set_transform(val_tf)

    else:
        # Normal single dataset
        dataset.set_transform(train_tf)
        val_ds.dataset.set_transform(val_tf)

    # Build loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Class count
    if isinstance(dataset, ConcatDataset):
        num_classes = len(dataset.datasets[0].classes)
    else:
        num_classes = len(dataset.classes)

    return train_loader, val_loader, num_classes