import argparse
import os
import json
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import (
    Compose, ToTensor, Normalize,
    RandomRotation, RandomPerspective,
    ColorJitter, GaussianBlur
)
import torch.nn.functional as F

from sklearn.model_selection import train_test_split


from protonet.data.emothaw_dataset import EMOTHAWDataset
from protonet.data.episodic_sampler import EpisodicSampler
from protonet.models.protonet import ProtoNet
from protonet.trainer.engine import ProtoEngine


###############################################
# Correct EMOTHAW task names (matching folder names)
###############################################
EMOTHAW_TASKS = ["pentagon", "house", "cdt", "cursive_writing", "words"]


###############################################
# Parse arguments
###############################################
parser = argparse.ArgumentParser()

parser.add_argument("--no_aug", action="store_true",
                    help="Disable data augmentation")

parser.add_argument("--task", type=str, default=None,
                    help="EMOTHAW task: pentagon, house, cdt, cursive_writing, words")

parser.add_argument("--data_root", type=str, default=None,
                    help="Custom dataset root instead of --task")

parser.add_argument("--train_all_tasks", action="store_true",
                    help="Train all EMOTHAW tasks sequentially")

parser.add_argument("--auto", action="store_true",
                    help="Automatically apply good defaults")

parser.add_argument("--way", type=int, default=3)
parser.add_argument("--shot", type=int, default=5)
parser.add_argument("--query", type=int, default=15)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--epochs", type=int, default=100)

parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

###############################################
# Split dataset into Train / Val / Test
###############################################
def split_emothaw(dataset, train_ratio=0.7, val_ratio=0.15):

    total_indices = list(range(len(dataset)))
    labels = [lbl for _, lbl in dataset.samples]

    # Train vs temp (val+test)
    train_idx, temp_idx = train_test_split(
        total_indices, test_size=(1 - train_ratio), stratify=labels
    )

    temp_labels = [labels[i] for i in temp_idx]
    val_size = val_ratio / (1 - train_ratio)   # e.g., 0.15 / 0.30 = 0.5

    # Val vs Test
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_size), stratify=temp_labels
    )

    return train_idx, val_idx, test_idx


###############################################
# TRANSFORMS
###############################################
if args.no_aug:
    print("\n*** RUNNING WITHOUT DATA AUGMENTATION ***\n")
    train_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5]),
    ])
    result_root = "results_no_aug"

else:
    print("\n*** RUNNING WITH DATA AUGMENTATION ***\n")
    train_transform = Compose([
        RandomRotation(10),
        RandomPerspective(distortion_scale=0.15, p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2),
        GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5]),
    ])
    result_root = "results_aug"


val_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5],
              std=[0.5, 0.5, 0.5])
])

test_transform = val_transform

def pad_collate(batch):
    """
    Pads images in a batch to the size of the largest H and W.
    Keeps aspect ratio (no resizing), avoids distortion.
    """

    images, labels = zip(*batch)

    # find max height and width
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded = []
    for img in images:
        h, w = img.shape[1], img.shape[2]
        pad_h = max_h - h
        pad_w = max_w - w

        img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
        padded.append(img)

    return torch.stack(padded), torch.tensor(labels)

###############################################
# Training a single task
###############################################
def train_single_task(task_name, data_root_override=None):

    # Determine dataset folder
    if data_root_override:
        data_root = data_root_override
    else:
        data_root = f"emothaw_tasks/{task_name}"

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Missing task folder: {data_root}")

    print(f"\n========== TRAINING TASK: {task_name} ==========")
    print(f"Dataset = {data_root}")

    # Dataset
    full_dataset = EMOTHAWDataset(data_root, transform=None)
    
    # Train/Val/Test split
    train_idx, val_idx, test_idx = split_emothaw(full_dataset)
    
    # Build subsets with transforms
    train_ds = Subset(EMOTHAWDataset(data_root, transform=train_transform), train_idx)
    val_ds   = Subset(EMOTHAWDataset(data_root, transform=val_transform), val_idx)
    test_ds  = Subset(EMOTHAWDataset(data_root, transform=test_transform), test_idx)
    
    # Labels for episodic sampling
    train_labels = [full_dataset.samples[i][1] for i in train_idx]
    val_labels   = [full_dataset.samples[i][1] for i in val_idx]
    test_labels  = [full_dataset.samples[i][1] for i in test_idx]

    # Samplers
    train_sampler = EpisodicSampler(
        labels=train_labels,
        n_way=args.way,
        k_shot=args.shot,
        q_query=args.query,
        episodes_per_epoch=args.episodes
    )
    
    # Validation episodes are smaller because validation split is small
    VAL_SHOT = 1
    VAL_QUERY = 3

    # Test episodes also smaller
    TEST_SHOT = 1
    TEST_QUERY = 5

    val_sampler = EpisodicSampler(
        labels=val_labels,
        n_way=args.way,
        k_shot=VAL_SHOT,
        q_query=VAL_QUERY,
        episodes_per_epoch=40
    )

    test_sampler = EpisodicSampler(
        labels=test_labels,
        n_way=args.way,
        k_shot=TEST_SHOT,
        q_query=TEST_QUERY,
        episodes_per_epoch=200
    )

    # Loaders
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=pad_collate)
    val_loader   = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=pad_collate)
    test_loader  = DataLoader(test_ds, batch_sampler=test_sampler, collate_fn=pad_collate)

    # Model
    model = ProtoNet(x_dim=3, hid_dim=64, z_dim=256)

    # Engine 
    engine = ProtoEngine(
        model=model,
        lr=1e-3,
        device=args.device
    )

    # Train
    engine.train_task(
        task_name=os.path.join(result_root, task_name),
        train_loader=train_loader,
        val_loader=val_loader,
        n_way=args.way,
        k_shot=args.shot,
        q_query=args.query,
        max_epochs=args.epochs
    )

    # TEST
    print(f"\n>>> Evaluating TEST set for task: {task_name.upper()}")
    test_loss, test_acc = engine.evaluate(
        test_loader=test_loader,
        n_way=args.way,
        k_shot=args.shot,
        q_query=args.query
    )

    print(f"\n[Test Results for {task_name.upper()}] Loss={test_loss:.4f}, Acc={test_acc:.4f}\n")

###############################################
# MODE 1 — Train ALL tasks
###############################################
if args.train_all_tasks:
    print("\n=== TRAINING ALL EMOTHAW TASKS ===")

    for t in EMOTHAW_TASKS:
        train_single_task(t)

    print("\nAll tasks finished.\n")
    exit(0)


###############################################
# MODE 2 — Train ONE task
###############################################
if args.task is not None:
    train_single_task(args.task)
    exit(0)

if args.data_root is not None:
    train_single_task("custom_task", data_root_override=args.data_root)
    exit(0)


###############################################
# Nothing specified
###############################################
print("ERROR: You must specify either --task or --data_root or --train_all_tasks")
