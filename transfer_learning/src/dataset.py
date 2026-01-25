import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
from torchvision.datasets import ImageFolder
import torch
import os
import pandas as pd

from src.pseudo_features import extract_pseudo_dynamic_features

PSEUDO_NUMERIC_FEATURES = [
    "path_length",
    "straightness",
    "dominant_angle",
    "direction_concentration",
    "width",
    "height",
    "aspect_ratio",
    "median_speed",
    "p95_speed",
]

def load_reverse_features(csv_path, id_col="id"):
    # ------------------------------------------------------------
    # Pseudo features reconstructed from the reverse model
    # ------------------------------------------------------------
    df = pd.read_csv(csv_path, sep=None, engine="python")
    
    df.columns = df.columns.str.strip()
    
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found. Available columns: {df.columns.tolist()}")
    
   # Convert ALL feature columns to numeric safely
    for col in PSEUDO_NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)

            df[col] = df[col].str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with invalid numeric values
    df = df.dropna(subset=PSEUDO_NUMERIC_FEATURES)

    features = {}
    for _, row in df.iterrows():
        image_id = str(row[id_col])
        feat = torch.tensor(
            row[PSEUDO_NUMERIC_FEATURES].values,
            dtype=torch.float32
        )
        features[image_id] = feat

    return features


# ------------------------------------------------------------
# Albumentations transforms
# ------------------------------------------------------------
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

    def __init__(self, root, pseudo_type="handcrafted", reverse_feat_path=None):
        super().__init__(root)
        
        self.pseudo_type = pseudo_type
        
        if self.pseudo_type == "reverse":
            if reverse_feat_path is None:
                raise ValueError("reverse_feat_path must be provided for reverse features")

            self.reverse_features = load_reverse_features(
                reverse_feat_path,
                id_col="id"  
            )

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.pseudo_type == "handcrafted":
            pseudo = extract_pseudo_dynamic_features(image)
            pseudo = torch.from_numpy(pseudo).float()

        elif self.pseudo_type == "reverse":
            # Extract ID from EMOTHAW image filename
            image_id = os.path.splitext(os.path.basename(path))[0]

            if image_id not in self.reverse_features:
                raise KeyError(f"Reverse features not found for {image_id}")

            pseudo = self.reverse_features[image_id].float()
            
        else:
            raise ValueError("Unknown pseudo_type")

        return image, pseudo, label


# ------------------------------------------------------------
# Wrapper that applies transforms AFTER random_split
# ------------------------------------------------------------
class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, pseudo,  label = self.subset[idx]  
        image = self.transform(image=image)["image"]
        return image, pseudo, label
    
    
    
def get_dataloaders(task, 
                    task_root, 
                    img_size, 
                    batch_size, 
                    num_workers, 
                    val_ratio,
                    pseudo_type="handcrafted",
                    reverse_feat_path=None):

    train_tf, val_tf = get_transforms(img_size)

    # ------------------------------
    # CASE 1: SINGLE TASK
    # ------------------------------
    if task != "all":
        task_root = os.path.join(task_root, task)
        print(f"Loading single task: {task_root}")
        dataset = AlbumentationsDataset(task_root,
                                        pseudo_type=pseudo_type,
                                        reverse_feat_path=reverse_feat_path)

    # ------------------------------
    # CASE 2: ALL TASKS COMBINED
    # ------------------------------
    else:
        print("Loading ALL tasks...")
        subfolders = sorted([
            f for f in os.listdir(task_root)
            if os.path.isdir(os.path.join(task_root, f))
        ])

        datasets = []
        for sub in subfolders:
            path = os.path.join(task_root, sub)
            print(" :", path)
            datasets.append(AlbumentationsDataset(path,
                                                 pseudo_type=pseudo_type,
                                                 reverse_feat_path=reverse_feat_path))

        dataset = ConcatDataset(datasets)
        
        
    # --------------------------------------------------------
    # TRAIN/VAL SPLIT
    # --------------------------------------------------------
    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    train_subset, val_subset = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
        
    # --------------------------------------------------------
    # APPLY TRANSFORMS 
    # --------------------------------------------------------   
    train_ds = TransformSubset(train_subset, train_tf)
    val_ds   = TransformSubset(val_subset, val_tf)
    
    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # --------------------------------------------------------
    # CLASS COUNT
    # --------------------------------------------------------
    if isinstance(dataset, ConcatDataset):
        num_classes = len(dataset.datasets[0].classes)
    else:
        num_classes = len(dataset.classes)

    return train_loader, val_loader, num_classes
