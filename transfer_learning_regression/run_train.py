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
from src.utils import save_checkpoint
from utils.plot_all import run_all_plots
from utils.plot_training import plot_metrics
from utils.visualize_aug import visualize_augmentations


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

    # Prepare output directory
    os.makedirs("outputs", exist_ok=True)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device → {device}")

    # --------------------------------------------------------
    # Load dataset(s)
    # --------------------------------------------------------
    train_loader, val_loader, _ = get_dataloaders(
        task=args.task,
        task_root=args.task_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        label_csv=args.label_csv
    )

    model = build_resnet18(output_dim=4, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    # Loss function
    criterion = nn.MSELoss()

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
    history = {"train_loss": [],  
               "val_loss": [],
               "lr": []}
    
    best_val_loss = float("inf")
    
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

        for images, pseudo, labels in tqdm(train_loader, desc="Training"):
            images = images.to(device)  
            pseudo = pseudo.to(device)
            labels = labels.to(device).float() 
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images, pseudo)
                loss = criterion(outputs, labels)

            # Backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()

        train_loss = train_loss_total / len(train_loader)

        # --------------------------------------------------------
        # VALIDATION PHASE
        # --------------------------------------------------------
        model.eval()
        val_loss_total = 0

        with torch.no_grad():
            for images, pseudo, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(device)  
                pseudo = pseudo.to(device)
                labels = labels.to(device).float()

                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images, pseudo)
                    loss = criterion(outputs, labels)

                val_loss_total += loss.item()

        val_loss = val_loss_total / len(val_loader)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        # --------------------------------------------------------
        # Logging + Scheduler Update
        # --------------------------------------------------------
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print("Current LR:", current_lr)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # --------------------------------------------------------
        # Save best model
        # --------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            filename = f"best_model_{args.task}_regression.pth"
            save_path = os.path.join("outputs", filename)

            save_checkpoint(model, save_path)
            print(f"Saved new BEST model : {filename}")

    # Save history file
    with open("outputs/history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining complete! History saved to outputs/history.json")
    
    print("\nRunning evaluation & plotting...")

    run_all_plots(
        task=args.task,
        task_dir=args.task_dir,
        model_path=os.path.join(
            "outputs", f"best_model_{args.task}_regression.pth"
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
    
    parser.add_argument("--task_dir", type=str, required=True,
                        help="Path to your dataset root folder containing class subdirectories.")
    
    parser.add_argument("--label_csv", type=str, required=True,
                    help="Path to CSV file with DASS labels")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for DataLoader.")

    args = parser.parse_args()

    run_train(args)