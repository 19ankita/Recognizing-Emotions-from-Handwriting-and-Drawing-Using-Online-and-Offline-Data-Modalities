import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
import torch
import os
import pandas as pd

from src.pseudo_features import extract_pseudo_dynamic_features

# ===================== PATH SETUP  =====================
base_dir = os.path.dirname(os.path.abspath(__file__))
features_dir = os.path.join(base_dir, "features")

# DASS scale constants
DASS_SCALE = torch.tensor([42.0, 42.0, 42.0, 126.0])

# ------------------------------------------------------------
# Labels for regression
# ------------------------------------------------------------
def load_dass_labels(csv_path):
    
    """
    Load DASS regression labels from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing sample IDs and DASS scores
        (depression, anxiety, stress, total).

    Returns
    -------
    label_map : dict
        Dictionary mapping each sample ID to a torch.FloatTensor of shape (4,).
    """

    df = pd.read_csv(csv_path)

    label_map = {}
    for _, row in df.iterrows():
        label_map[row["id"]] = torch.tensor(
            [
                row["depression"],
                row["anxiety"],
                row["stress"],
                row["total"]
            ],
            dtype=torch.float32
        )
    return label_map


# ------------------------------------------------------------
# Albumentations transforms
# ------------------------------------------------------------
def get_transforms(img_size):
    
    """
    Define training and validation image transformations using Albumentations.
    
    Parameters
    ----------
    img_size : int
        Target image height and width.

    Returns
    -------
    train_tf : albumentations.Compose
        Training transform with strong data augmentation.
    val_tf : albumentations.Compose
        Validation transform with resizing and normalization.
    """

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Strong augmentation for training
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
# Albumentations wrapper for Dataset
# ------------------------------------------------------------
class AlbumentationsDataset(Dataset):
    
    """
    Dataset for loading handwriting images with associated pseudo-dynamic
    features and DASS regression labels.

    Images are loaded in grayscale, converted to RGB, and returned as NumPy
    arrays. Pseudo-dynamic features are extracted from the unaugmented image.

    Parameters
    ----------
    root : str
        Root directory containing class-wise subfolders of PNG images.
    label_map : dict
        Mapping from image ID to DASS label tensor
        (depression, anxiety, stress, total).

    Returns
    -------
    image : numpy.ndarray
        Handwriting image of shape (H, W, 3).
    pseudo : torch.Tensor
        Pseudo-dynamic feature vector.
    label : torch.Tensor
        DASS regression targets of shape (4,).
    """

    def __init__(self, root, label_map, normalize_labels=True):
        self.label_map = label_map
        self.normalize_labels = normalize_labels
        self.samples = []
        
        for class_dir in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            for fname in sorted(os.listdir(class_path)):
                if not fname.endswith(".png"):
                    continue
                
                image_id = os.path.splitext(fname)[0]
                
                if image_id in self.label_map:
                    self.samples.append(os.path.join(class_path, fname))
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]

       # ---------------- IMAGE ----------------
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # ---------------- PSEUDO-DYNAMIC FEATURES ----------------
        # Extracted BEFORE augmentation
        pseudo = extract_pseudo_dynamic_features(image)
        pseudo = torch.tensor(pseudo, dtype=torch.float32)
        
        # ---------------- LABEL ----------------
        image_id = os.path.splitext(os.path.basename(path))[0]
        label = self.label_map[image_id]
        
        if self.normalize_labels:
            label = label / DASS_SCALE

        return image, pseudo, label


# ------------------------------------------------------------
# Wrapper that applies transforms AFTER random_split
# ------------------------------------------------------------
class TransformSubset(Dataset):
    """
    Wrapper dataset that applies Albumentations transforms to images only.

    Parameters
    ----------
    subset : torch.utils.data.Dataset
        Dataset returning (image, pseudo, label).
    transform : albumentations.Compose
        Image transformation pipeline.

    Returns
    -------
    tuple
        Transformed image, unchanged pseudo-features, and label.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, pseudo, label = self.subset[idx]  
        image = self.transform(image=image)["image"]
        return image, pseudo, label
    
    
    
def get_dataloaders(task, task_root, img_size, batch_size, num_workers, val_ratio, label_csv):
    
    """
    Create training and validation DataLoaders for handwriting emotion regression.

    Supports loading a single task or combining all tasks into one dataset.
    Applies strong augmentation to training data and normalization to validation data.

    Parameters
    ----------
    task : str
        Task name or "all" to combine all task subfolders.
    task_root : str
        Root directory containing task subfolders.
    img_size : int
        Target image height and width.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of DataLoader worker processes.
    val_ratio : float
        Fraction of data used for validation.
    label_csv : str
        Path to CSV file containing DASS labels.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data.
    num_classes : None
        Placeholder for compatibility with classification pipelines.
    """

    train_tf, val_tf = get_transforms(img_size)
    label_map = load_dass_labels(label_csv)

    # ------------------------------
    # CASE 1: SINGLE TASK
    # ------------------------------
    if task != "all":
        task_root = os.path.join(task_root, task)
        print(f"Loading single task: {task_root}")
        dataset = AlbumentationsDataset(task_root, label_map, normalize_labels=True)

    # ------------------------------
    # CASE 2: ALL TASKS COMBINED
    # ------------------------------
    else:
        print("Loading ALL tasks...")
        datasets = []
        subfolders = sorted([
            f for f in os.listdir(task_root)
            if os.path.isdir(os.path.join(task_root, f))
        ])
        
        for sub in subfolders:
            path = os.path.join(task_root, sub)
            print(" :", path)
            datasets.append(AlbumentationsDataset(path, label_map, normalize_labels=True))

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
    train_loader = DataLoader(train_ds, 
                              batch_size=batch_size,
                              shuffle=True,  
                              num_workers=num_workers,
                              pin_memory=True)
    
    val_loader   = DataLoader(val_ds, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=num_workers,
                              pin_memory=True)


    return train_loader, val_loader, None
