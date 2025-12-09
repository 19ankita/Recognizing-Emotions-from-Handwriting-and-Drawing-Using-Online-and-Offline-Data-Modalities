import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np

class Cutout(object):
    def __init__(self, n_holes=1, length=30):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size[1], img.size[0]
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask


def get_transforms(cfg):
    img_size = cfg["img_size"]

    # train_tf = transforms.Compose([
    #     transforms.RandomResizedCrop(img_size, scale=(0.90, 1.0)),
        
    #     transforms.RandomHorizontalFlip(p=0.2),
        
    #     transforms.RandomRotation(5),
        
    #     #transforms.RandomAffine(degrees=0, 
    #                             #translate=(0.08, 0.08),
    #                             #shear=5),

    #     transforms.ColorJitter(brightness=0.10,
    #                            contrast=0.10),
        
    #     transforms.ToTensor(),
        
    #     Cutout(n_holes=1, length=12),
        
    #     transforms.Normalize([0.485, 0.456, 0.406],
    #                          [0.229, 0.224, 0.225])
    # ])
    
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
])


    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.05)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    return train_tf, val_tf


def get_dataloaders(cfg):
    train_tf, val_tf = get_transforms(cfg)

    # Load dataset WITHOUT transform
    full_dataset = ImageFolder(cfg["task_dir"], transform=None)

    val_ratio = cfg.get("val_ratio", 0.2)  # default 20% validation
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    # Split dataset randomly
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # reproducible split
    )

    # Apply transforms to train and validation set
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, 
                              batch_size=cfg["batch_size"],
                              shuffle=True, 
                              num_workers=cfg["num_workers"])

    val_loader = DataLoader(val_ds, 
                            batch_size=cfg["batch_size"],
                            shuffle=False, 
                            num_workers=cfg["num_workers"])

    # Number of classes:
    num_classes = len(full_dataset.classes)

    return train_loader, val_loader, num_classes
