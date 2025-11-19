import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from protonet.data.emothaw_dataset import EMOTHAWDataset
from protonet.data.episodic_sampler import EpisodicSampler
from protonet.models.protonet import ProtoNet
from protonet.trainer.engine import ProtoEngine


###############################################
# Correct EMOTHAW task names (matching folder names)
###############################################
EMOTHAW_TASKS = ["pentagon", "house", "cdt", "cursive_writing", "words"]

# ============================================================
# FIXED, CLEAN IMAGE TRANSFORMS (NO AUGMENTATION)
# ============================================================
def get_transforms():
    return Compose([
        Resize((224, 224)),              # FIXED SIZE   << important
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5])
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

parser.add_argument("--way", type=int, default=3)
parser.add_argument("--shot", type=int, default=5)
parser.add_argument("--query", type=int, default=15)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--epochs", type=int, default=100)

parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()


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
    
    # Transform
    transform = get_transforms()

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
        k_shot=1,
        q_query=5,
        episodes_per_epoch=40
    )

    test_sampler = EpisodicSampler(
        labels=labels_all,
        n_way=args.way,
        k_shot=1,
        q_query=5,
        episodes_per_epoch=200
    )

    # Loaders
    train_loader = DataLoader(full_dataset, 
                              batch_sampler=train_sampler) 

    val_loader   = DataLoader(full_dataset, 
                              batch_sampler=val_sampler) 

    test_loader  = DataLoader(full_dataset, 
                              batch_sampler=test_sampler) 

    # Model
    model = ProtoNet(x_dim=3, hid_dim=64, z_dim=128)

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