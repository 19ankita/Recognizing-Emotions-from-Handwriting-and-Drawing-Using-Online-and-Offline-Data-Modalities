import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os
import math
from pathlib import Path

from src.dataset import get_dataloaders
from src.model import build_resnet18
from src.utils import accuracy, save_checkpoint
from utils.helper import get_class_names_from_task
from utils.plot_all import run_all_plots
from utils.plot_training import plot_metrics
from utils.visualize_aug import visualize_augmentations
from src.pseudo_features import extract_pseudo_dynamic_features
from utils.visualize_pseudodynamic_features import visualize_single_image


torch.backends.cudnn.benchmark = True

# ------------------------------------------------------------
# Warmup + Cosine LR Scheduler
# ------------------------------------------------------------
def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        # Warmup phase
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        # Cosine decay
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    
# ------------------------------------------------------------
# Training Function
# ------------------------------------------------------------
def run_train(args):
    
    if args.task == "all" and args.pseudo_type == "reverse":
        raise ValueError(
            "pseudo_type='reverse' is not supported with task='all' "
            "(reverse features are task-specific)"
        )

    # Prepare output directory
    os.makedirs("outputs", exist_ok=True)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device → {device}")
    
    reverse_feat_path = None
    if args.pseudo_type == "reverse":
        reverse_feat_path = f"pseudo_features/{args.task}_pseudo_features.csv"

    # --------------------------------------------------------
    # Load dataset(s)
    # --------------------------------------------------------
    train_loader, val_loader, num_classes = get_dataloaders(
        task=args.task,
        task_root=args.task_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        pseudo_type=args.pseudo_type,
        reverse_feat_path=reverse_feat_path
    )
    
    class_names = get_class_names_from_task(args.task_dir, args.task)


    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------
    print(f"\n>>> Building model: {args.model}")
    if args.model == "resnet18":
        model = build_resnet18(num_classes, freeze_backbone=args.freeze_backbone)
    else:
        raise ValueError("Unknown model: choose resnet18 or resnet50")

    model = model.to(device)


    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer 
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    # Scheduler 
    scheduler = get_scheduler(
        optimizer,
        warmup_epochs=2,
        total_epochs=args.epochs
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # Training history storage
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    best_acc = 0

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(args.epochs):
        print(f"\n==== Epoch {epoch+1}/{args.epochs} ====")

        # --------------------------------------------------------
        # TRAINING PHASE
        # --------------------------------------------------------
        model.train()
        train_loss_total = 0
        train_acc_total = 0

        for images, pseudo, labels in tqdm(train_loader, desc="Training"):
            images = images.to(device)  
            pseudo = pseudo.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images, pseudo)
                loss = criterion(outputs, labels)

            # Backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()
            train_acc_total += accuracy(outputs, labels)

        train_loss = train_loss_total / len(train_loader)
        train_acc = train_acc_total / len(train_loader)

        # --------------------------------------------------------
        # VALIDATION PHASE
        # --------------------------------------------------------
        model.eval()
        val_loss_total = 0
        val_acc_total = 0

        with torch.no_grad():
            for images, pseudo, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(device)  
                pseudo = pseudo.to(device)
                labels = labels.to(device)

                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images, pseudo)
                    loss = criterion(outputs, labels)

                val_loss_total += loss.item()
                val_acc_total += accuracy(outputs, labels)

        val_loss = val_loss_total / len(val_loader)
        val_acc = val_acc_total / len(val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # --------------------------------------------------------
        # Logging + Scheduler Update
        # --------------------------------------------------------
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print("Current LR:", current_lr)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # --------------------------------------------------------
        # Save best model
        # --------------------------------------------------------
        if val_acc > best_acc:
            best_acc = val_acc
            
            filename = f"best_model_{args.task}_{args.model}.pth"
            save_path = os.path.join("outputs", filename)

            save_checkpoint(model, save_path)
            print(f"Saved new BEST model : {filename}")
                
            print("Saved new BEST model")
            
    # ------------------------------------------------------------
    # Pseudo-Dynamic Feature Visualization (single sample)
    # ------------------------------------------------------------
    pseudo_vis_dir = "outputs/pseudo_features"
    os.makedirs(pseudo_vis_dir, exist_ok=True)
    
    for class_name in class_names:
        img_dir = Path(args.task_dir) / args.task / class_name

        images = list(img_dir.glob("*.png"))

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    sample_img_path = next(img_dir.glob("*.png"))

    if args.pseudo_type == "handcrafted":
        visualize_single_image(
            image_path=sample_img_path,
            save_dir=pseudo_vis_dir
        )


    print("Pseudo-dynamic feature visualization saved.")


    # Save history file
    with open("outputs/history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining complete! History saved to outputs/history.json")
    
    print("\nRunning evaluation & plotting...")

    run_all_plots(
        task=args.task,
        task_dir=args.task_dir,
        model_path=os.path.join(
            "outputs", f"best_model_{args.task}_{args.model}.pth"
        ),
        history_path="outputs/history.json",
        output_dir="outputs"
    )
    
    plot_metrics("outputs/history.json")
    
    visualize_augmentations(args.task, args.task_dir, args.img_size)
    
    print("\nRunning stress diagnostics...")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True,
                        help="Select task: each task name (folder) or 'all' for all tasks combined")

    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"])
    
    parser.add_argument("--task_dir", type=str, required=True,
                        help="Path to your dataset root folder containing class subdirectories.")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for DataLoader.")
    
    parser.add_argument("--pseudo_type", type=str, default="handcrafted", choices=["handcrafted", "reverse"],
                        help="Type of pseudo features to use")

    args = parser.parse_args()

    run_train(args)