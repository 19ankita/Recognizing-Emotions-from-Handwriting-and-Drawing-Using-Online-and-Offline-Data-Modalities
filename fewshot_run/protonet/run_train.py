import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize,
    RandomRotation, ColorJitter, RandomAffine
)

from protonet.data.emothaw_dataset import EMOTHAWDataset
from protonet.data.episodic_sampler import EpisodicSampler
from protonet.models.protonet import ProtoNet
from protonet.models.backbones import Conv6Backbone, Conv8Backbone, ResNet18Encoder
from protonet.trainer.engine import ProtoEngine


###############################################
# Correct EMOTHAW task names (matching folder names)
###############################################
EMOTHAW_TASKS = ["pentagon", "house", "cdt", "cursive_writing", "words"]

# ------------------------------------------------------------
# Compute dataset mean / std (cached)
# ------------------------------------------------------------
def compute_dataset_stats(dataset_root):
    stats_file = os.path.join(dataset_root, "norm_stats.pt")
    if os.path.exists(stats_file):
        return torch.load(stats_file)

    print("Computing dataset mean/std... (one-time)")
    dataset = EMOTHAWDataset(dataset_root, transform=Compose([Resize((84, 84)), ToTensor()]))
    imgs = torch.stack([img for img, _ in dataset], dim=0)
    mean = imgs.mean(dim=[0, 2, 3])
    std = imgs.std(dim=[0, 2, 3])
    torch.save({"mean": mean, "std": std}, stats_file)
    return {"mean": mean, "std": std}

# ============================================================
# FIXED, CLEAN IMAGE TRANSFORMS (NO AUGMENTATION)
# ============================================================
def get_transforms(mean, std, augment=False):

    aug = []
    if augment:
        aug = [
            RandomRotation(10),
            RandomAffine(10, translate=(0.05, 0.05)),
            ColorJitter(brightness=0.1, contrast=0.1)
        ]

    return Compose([
        Resize((84, 84)),     # << CHANGED FROM 224→84
        *aug,
        ToTensor(),
        Normalize(mean.tolist(), std.tolist())
    ])


###############################################
# Parse arguments
###############################################
parser = argparse.ArgumentParser()

parser.add_argument("--task", type=str, default=None,
                    help="EMOTHAW task: pentagon, house, cdt, cursive_writing, words")

parser.add_argument("--data_root", type=str, default=None,
                    help="Custom dataset root instead of --task")

parser.add_argument("--train_all_tasks", action="store_true",
                    help="Train all EMOTHAW tasks sequentially")

parser.add_argument("--auto", action="store_true",
                    help="Automatically apply good defaults")

parser.add_argument("--encoder", type=str, default="proto4",
                    choices=["proto4", "conv6", "conv8", "resnet18"])

parser.add_argument("--way", type=int, default=3)
parser.add_argument("--shot", type=int, default=5)
parser.add_argument("--query", type=int, default=15)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--epochs", type=int, default=100)

parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

# ------------------------------------------------------------
# Build encoder
# ------------------------------------------------------------
def build_encoder(encoder_name):
    if encoder_name == "proto4":
        return ProtoNet(x_dim=3, hid_dim=64, z_dim=128)
    if encoder_name == "conv6":
        return Conv6Backbone()
    if encoder_name == "conv8":
        return Conv8Backbone()
    if encoder_name == "resnet18":
        return ResNet18Encoder(output_dim=128)

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
    
    # Compute mean/std once
    stats = compute_dataset_stats(data_root)
    mean, std = stats["mean"], stats["std"]

    transform = get_transforms(mean, std, augment=args.augment)

    # Load dataset FULL
    full_dataset = EMOTHAWDataset(data_root, transform=transform)

    # Labels for entire dataset
    labels_all = [lbl for _, lbl in full_dataset.samples]
    
    print("Unique labels in dataset:", sorted(set(labels_all)))

    # Samplers
    train_sampler = EpisodicSampler(
        labels=labels_all,
        n_way=args.way,
        k_shot=args.shot,
        q_query=args.query,
        episodes_per_epoch=args.episodes
    )

    val_sampler = EpisodicSampler(
        labels=labels_all,
        n_way=args.way,
        k_shot=args.shot,
        q_query=args.query,
        episodes_per_epoch=40
    )

    test_sampler = EpisodicSampler(
        labels=labels_all,
        n_way=args.way,
        k_shot=args.shot,
        q_query=args.query,
        episodes_per_epoch=200
    )

    # Loaders
    train_loader = DataLoader(full_dataset, 
                              batch_sampler=train_sampler, 
                              shuffle=False) 

    val_loader   = DataLoader(full_dataset, 
                              batch_sampler=val_sampler) 

    test_loader  = DataLoader(full_dataset, 
                              batch_sampler=test_sampler) 

    # Model
    model = build_encoder(args.encoder)

    # Engine 
    engine = ProtoEngine(
        model=model,
        lr=1e-3,
        device=args.device
    )

    # TRAIN -------------------------------------------------------------
    history, exp_dir = engine.train_task(
        task_name=task_name,
        train_loader=train_loader,
        val_loader=val_loader,
        n_way=args.way,
        k_shot=args.shot,
        q_query=args.query,
        max_epochs=args.epochs
    )
    # TEST --------------------------------------------------------------
    print(f"\n>>> Running TEST episodes for {task_name.upper()}...")
    engine.evaluate(
        test_loader=test_loader,
        n_way=args.way,
        k_shot=1,
        q_query=5
    )

train_single_task(args.task)