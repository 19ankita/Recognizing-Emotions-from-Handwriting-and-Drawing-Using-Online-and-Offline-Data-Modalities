import argparse
import os
import json
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


###############################################
# Parse arguments
###############################################
parser = argparse.ArgumentParser()

parser.add_argument("--task", type=str, default=None,
                    help="EMOTHAW task: pentagon, house, cdt, cursive_writing, words")

parser.add_argument("--data_root", type=str, default=None,
                    help="Custom dataset root instead of --task")

parser.add_argument("--config", type=str, default=None,
                    help="JSON file mapping tasks to custom dataset paths")

parser.add_argument("--train_all_tasks", action="store_true",
                    help="Train all EMOTHAW tasks sequentially")

parser.add_argument("--auto", action="store_true",
                    help="Automatically apply good defaults")

parser.add_argument("--way", type=int, default=3)
parser.add_argument("--shot", type=int, default=5)
parser.add_argument("--query", type=int, default=10)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--epochs", type=int, default=10)

parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()


###############################################
# Load config (optional)
###############################################
task_map = {}
if args.config is not None:
    if not os.path.isfile(args.config):
        raise FileNotFoundError(args.config)
    with open(args.config, "r") as f:
        task_map = json.load(f)


###############################################
# Training a single task
###############################################
def train_single_task(task_name, data_root_override=None):

    # Determine dataset path
    if data_root_override is not None:
        data_root = data_root_override
    elif task_name in task_map:
        data_root = task_map[task_name]
    else:
        data_root = f"emothaw_tasks/{task_name}"

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Missing task folder: {data_root}")

    print(f"\n========== TRAINING TASK: {task_name} ==========")
    print(f"Dataset = {data_root}")

    # Transform
    transform = Compose([
        Resize((128, 128)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5])
    ])

    # Dataset
    dataset = EMOTHAWDataset(data_root, transform=transform)
    labels = [lbl for _, lbl in dataset.samples]

    sampler = EpisodicSampler(
        labels=labels,
        n_way=args.way,
        k_shot=args.shot,
        q_query=args.query,
        episodes_per_epoch=args.episodes
    )

    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    # Validation = same dataset
    val_loader = loader

    # Model
    model = ProtoNet(x_dim=3, hid_dim=64, z_dim=64)

    # Engine (new engine no longer needs log_dir or TB init here)
    engine = ProtoEngine(
        model=model,
        lr=1e-3,
        device=args.device
    )

    # Call the new training function name:
    engine.train_task(
        task_name=task_name,
        train_loader=loader,
        val_loader=val_loader,
        n_way=args.way,
        k_shot=args.shot,
        q_query=args.query,
        max_epochs=args.epochs
    )


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
