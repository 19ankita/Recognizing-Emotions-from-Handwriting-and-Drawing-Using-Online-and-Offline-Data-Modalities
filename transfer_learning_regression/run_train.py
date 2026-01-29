import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os
import math

from src.dataset import get_dataloaders
from src.model import build_resnet18
from src.utils import save_checkpoint
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from utils.plot_training_csv import plot_training_from_csv
import csv


torch.backends.cudnn.benchmark = True
DASS_NAMES = ["Depression", "Anxiety", "Stress", "Total"]

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
    print(f"Using device: {device}")

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
    
    # ------------------------------------------------------------
    # CSV setup
    # ------------------------------------------------------------
    csv_path = os.path.join("outputs", "training_metrics.csv")
    best_csv = os.path.join("outputs", "best_epoch_summary.csv")
    
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_mse",
            "val_mse",
            "val_rmse",
            "val_r2",
            "rmse_dep", "rmse_anx", "rmse_str", "rmse_tot",
            "r2_dep", "r2_anx", "r2_str", "r2_tot",
            "lr"
        ])
        
    best_val_r2 = -float("inf")
    best_epoch_row = None
        
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

        train_mse = train_loss_total / len(train_loader)

        # --------------------------------------------------------
        # VALIDATION PHASE
        # --------------------------------------------------------
        model.eval()
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, pseudo, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(device)  
                pseudo = pseudo.to(device)
                labels = labels.to(device).float()

                with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    outputs = model(images, pseudo)

                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        # Getting a summary of teh run    
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Overall metrics
        val_mse = mean_squared_error(all_labels, all_preds)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score(all_labels, all_preds, multioutput="uniform_average")     
        
        rmse_dims, r2_dims = [], []
        for i in range(4):
            rmse_dims.append(np.sqrt(mean_squared_error(all_labels[:, i], all_preds[:, i])))
            r2_dims.append(r2_score(all_labels[:, i], all_preds[:, i]))

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()   
        
        # ---------------- LOG ----------------
        print(f"Train MSE: {train_mse:.4f}")
        print(f"Val   MSE: {val_mse:.4f}")
        print(f"Val  RMSE: {val_rmse:.4f}")
        print(f"Val    R²: {val_r2:.4f}")

        for name, rmse_i, r2_i in zip(DASS_NAMES, rmse_dims, r2_dims):
            print(f"{name:<12} | RMSE: {rmse_i:.3f} | R²: {r2_i:.3f}")
            
        row = [
            epoch + 1,
            train_mse,
            val_mse,
            val_rmse,
            val_r2,
            *rmse_dims,
            *r2_dims,
            current_lr
        ]

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # ---------------- BEST MODEL (by val R²) ----------------
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch_row = row
            save_checkpoint(model, os.path.join("outputs", f"best_model_{args.task}_regression.pth"))


    # ------------------------------------------------------------
    # Save best epoch summary
    # ------------------------------------------------------------
    with open(best_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_mse", "val_mse", "val_rmse", "val_r2",
            "rmse_dep", "rmse_anx", "rmse_str", "rmse_tot",
            "r2_dep", "r2_anx", "r2_str", "r2_tot",
            "lr"
        ])
        writer.writerow(best_epoch_row)

    print(f"\nBest epoch summary saved to {best_csv}")
    
    print("\n plotting...")
    plot_training_from_csv(
        csv_path="outputs/training_metrics.csv",
        output_dir="outputs"
    )
    print(f"\nTraining plots saved to {output_dir}")
        
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