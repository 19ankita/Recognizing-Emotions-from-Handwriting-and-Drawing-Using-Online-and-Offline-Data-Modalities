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

def resolve_task_csv(task_root, task_name):

    """
    Resolve the path to the task-specific DASS annotation CSV.

    Input:
        task_root (str): Root directory containing all handwriting tasks.
        task_name (str): Name of the handwriting task.

    Output:
        str: Full path to the corresponding task-specific DASS CSV file.
    """

    return os.path.join(
        task_root,
        "features",
        f"{task_name}_with_dass.csv"
    )

# ------------------------------------------------------------
# Depression score → class mapping (DASS-21)
# ------------------------------------------------------------
def score_to_class(score, state):

    """
    Convert a DASS-21 score into a categorical severity class.

    The conversion follows the official DASS-21 severity thresholds and
    produces a five-class label corresponding to increasing symptom severity.

    Severity classes:
        0 : Normal
        1 : Mild
        2 : Moderate
        3 : Severe
        4 : Extremely Severe

    Parameters
    ----------
    score : float or int
        Raw DASS-21 score for a given emotional state.
    state : str
        Emotional dimension to be mapped. Must be one of
        {"depression", "anxiety", "stress"}.

    Returns
    -------
    int
        Severity class label in the range [0, 4].
    """


    if state == "depression":
        thresholds = [9, 13, 20, 27]
    elif state == "anxiety":
        thresholds = [7, 9, 14, 19]
    elif state == "stress":
        thresholds = [14, 18, 25, 33]
    else:
        raise ValueError(f"Unknown task: {state}")

    if score <= thresholds[0]:
        return 0
    elif score <= thresholds[1]:
        return 1
    elif score <= thresholds[2]:
        return 2
    elif score <= thresholds[3]:
        return 3
    else:
        return 4

# ------------------------------------------------------------
# Load depression classification labels
# ------------------------------------------------------------
def load_dass_labels(csv_path, state):

    """
    Load DASS-21 scores from a CSV file and convert them into severity classes.

    Each sample is assigned a categorical label based on the DASS-21 severity
    thresholds corresponding to the selected emotional state.

    Parameters
    ----------
    csv_path : str
        Path to the task-specific CSV file containing sample identifiers
        and DASS-21 scores.
    state : str
        Emotional dimension to be used as the classification target.
        Must be one of {"depression", "anxiety", "stress"}.

    Returns
    -------
    dict
        Dictionary mapping sample IDs to integer severity class labels (0–4).
    """


    df = pd.read_csv(csv_path)
    
    if state not in ["depression", "anxiety", "stress"]:
        raise ValueError(f"Invalid task: {state}")

    label_map = {}
    for _, row in df.iterrows():
        label_map[row["class"]] = score_to_class(row[state], state)

    return label_map


# ------------------------------------------------------------
# Albumentations transforms
# ------------------------------------------------------------
def get_transforms(img_size):
    
    """
    Define image preprocessing and augmentation pipelines.

    Strong data augmentation is applied during training to improve robustness,
    while validation images undergo only resizing and normalization.

    Parameters
    ----------
    img_size : int
        Target height and width of the output images.

    Returns
    -------
    train_tf : albumentations.Compose
        Training transformation pipeline with geometric and photometric
        augmentations.
    val_tf : albumentations.Compose
        Validation transformation pipeline with resizing and normalization.
    """


    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Strong augmentation for training
    train_tf = A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.10, rotate_limit=8, p=0.8
        ),
        A.ElasticTransform(
            alpha=10, sigma=50, p=0.5
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
    Dataset for handwriting-based emotion classification using images and
    pseudo-dynamic features.

    Each sample consists of a grayscale handwriting image, converted to RGB,
    a vector of pseudo-dynamic features extracted from the unaugmented image,
    and a categorical DASS-21 severity label.

    Parameters
    ----------
    root : str
        Root directory containing handwriting images organized in
        class-wise subfolders.
    label_map : dict
        Mapping from image identifiers to integer DASS-21 severity labels.

    Returns
    -------
    image : numpy.ndarray
        Handwriting image of shape (H, W, 3).
    pseudo : torch.Tensor
        Pseudo-dynamic feature vector extracted from the image.
    label : int
        DASS-21 severity class label in the range [0, 4].
    """


    def __init__(self, root, label_map):
        self.label_map = label_map
        self.samples = []
        
        for class_dir in os.listdir(root):
            class_path = os.path.join(root, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            for fname in os.listdir(class_path):
                if not fname.endswith(".png"):
                    continue
                
                image_id = os.path.splitext(fname)[0]
                
                if image_id in self.label_map:
                    self.samples.append(os.path.join(class_path, fname))
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # extract pseudo dynamic features BEFORE augmentation
        pseudo = extract_pseudo_dynamic_features(image)
        
        # image_id from filename
        image_id = os.path.splitext(os.path.basename(path))[0]
        
        label = self.label_map[image_id]

        return image, pseudo, label


# ------------------------------------------------------------
# Wrapper that applies transforms AFTER random_split
# ------------------------------------------------------------
class TransformSubset(Dataset):

    """
    Dataset wrapper that applies image transformations after dataset splitting.

    This wrapper ensures that data augmentation is applied only to the image
    component of each sample, while pseudo-dynamic features and labels remain
    unchanged.

    Parameters
    ----------
    subset : torch.utils.data.Dataset
        Subset of the dataset returning (image, pseudo, label).
    transform : albumentations.Compose
        Image transformation pipeline.

    Returns
    -------
    tuple
        Transformed image tensor, unchanged pseudo-dynamic features,
        and corresponding label.
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
    
    
    
def get_dataloaders(task, task_root, img_size, batch_size, num_workers, val_ratio, label_csv, state):
    
    """
    Create training and validation DataLoaders for handwriting emotion
    severity classification.

    The function supports both single-task training and joint training across
    all handwriting tasks. For single-task runs, a task-specific annotation
    CSV is used. When all tasks are combined, each task is paired with its
    corresponding annotation file before concatenation.

    Strong data augmentation is applied to the training set, while the
    validation set is only resized and normalized.

    Parameters
    ----------
    task : str
        Name of the handwriting task or "all" to combine all available tasks.
    task_root : str
        Root directory containing task subfolders.
    img_size : int
        Target image height and width.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of worker processes for data loading.
    val_ratio : float
        Fraction of the dataset reserved for validation.
    label_csv : str or None
        Path to the task-specific CSV file containing DASS-21 labels.
        Must be provided when task != "all".
    state : str
        Emotional dimension to be classified. One of
        {"depression", "anxiety", "stress"}.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    None
        Placeholder for compatibility with other pipelines.
    """

    train_tf, val_tf = get_transforms(img_size)
    
    if task != "all" and label_csv is None:
        raise ValueError("label_csv must be provided for single-task loading")

    # ------------------------------
    # CASE 1: SINGLE TASK
    # ------------------------------
    if task != "all":
        task_root = os.path.join(task_root, task)
        print(f"Loading single task: {task_root}")
        label_map = load_dass_labels(label_csv, state)
        dataset = AlbumentationsDataset(task_root, label_map)

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
            csv_path = resolve_task_csv(task_root, sub)
            
            print(f" : {path} | labels: {csv_path}")
            
            label_map = load_dass_labels(csv_path, state)
            datasets.append(AlbumentationsDataset(path, label_map))

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
                              num_workers=num_workers)
    
    val_loader   = DataLoader(val_ds, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=num_workers)


    return train_loader, val_loader, None
