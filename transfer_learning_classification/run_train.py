import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import math

from src.dataset import get_dataloaders
from src.model import build_resnet18
from src.utils import save_checkpoint
from sklearn.metrics import accuracy_score
import numpy as np
from utils.plot_training import plot_metrics
from utils.plot_matrices import plot_confusion_matrix, save_classification_report
import csv


torch.backends.cudnn.benchmark = True
CLASS_NAMES = [
    "Normal",
    "Mild",
    "Moderate",
    "Severe",
    "Extremely Severe"
]
NUM_CLASSES = 5

# ------------------------------------------------------------
# Warmup + Cosine LR Scheduler
# ------------------------------------------------------------
def get_scheduler(optimizer, warmup_epochs, total_epochs):
    
    """
    Create a learning rate scheduler with linear warmup followed by cosine decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate will be scheduled.
    warmup_epochs : int
        Number of epochs for linear learning rate warmup.
    total_epochs : int
        Total number of training epochs.

    Returns
    -------
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Learning rate scheduler implementing warmup and cosine decay.
    """

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
    
    """
    Train and evaluate a multi-output regression model for emotion recognition
    from handwriting data.

    This function handles dataset loading, model initialization, training with
    MSE loss and mixed-precision optimization, validation using RMSE and R²
    metrics, and learning rate scheduling with linear warmup followed by cosine
    decay. Per-epoch metrics are logged to CSV files, and the best-performing
    model is saved based on validation R².

    Parameters
    ----------
    args : argparse.Namespace
        Training configuration and hyperparameters, including dataset paths,
        model settings, optimizer parameters, and training options.

    Returns
    -------
    None
        Training results are saved to disk (model checkpoints, CSV logs, plots).
    """

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

    # model
    model = build_resnet18(output_dim=4, freeze_backbone=args.freeze_backbone)
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
    
    # ------------------------------------------------------------
    # CSV setup
    # ------------------------------------------------------------
    csv_path = os.path.join("outputs", "training_metrics.csv")
    best_csv = os.path.join("outputs", "best_epoch_summary.csv")
    
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "train_acc",
            "val_acc",
            "lr"
        ])
        
    best_val_acc = 0.0
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
        train_preds, train_labels = [], []

        for images, pseudo, labels in tqdm(train_loader, desc="Training"):
            images = images.to(device)  
            pseudo = pseudo.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # mixed precision training
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images, pseudo) # model prediction
                loss = criterion(outputs, labels)

            # Backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer) #unscaling
            scaler.update()

            train_loss_total += loss.item()
            train_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.append(labels.cpu().numpy())
            
        train_preds = np.concatenate(train_preds)
        train_labels = np.concatenate(train_labels)    

        train_loss = train_loss_total / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)

        # --------------------------------------------------------
        # VALIDATION PHASE
        # --------------------------------------------------------
        model.eval()
        
        val_loss_total = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for images, pseudo, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(device)  
                pseudo = pseudo.to(device)
                labels = labels.to(device)

                with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    outputs = model(images, pseudo)
                    loss = criterion(outputs, labels)

                val_loss_total += loss.item()
                val_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)

        val_loss = val_loss_total / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()   
        
        # ---------------- LOG ----------------
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
            
        row = [
            epoch + 1,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            current_lr
        ]

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # ---------------- BEST MODEL (by val R²) ----------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch_row = row
            save_checkpoint(
                model,
                os.path.join("outputs", f"best_model_{args.task}_depression_cls.pth")
            )


    # ------------------------------------------------------------
    # Save best epoch summary
    # ------------------------------------------------------------
    with open(best_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"
        ])
        writer.writerow(best_epoch_row)

    print(f"\nBest epoch summary saved to {best_csv}")
    
    print("\n plotting...")
    
    # Confusion matrix
    plot_confusion_matrix(val_labels, val_preds)
    
    # Loss and accuracy curves
    plot_metrics(
        csv_path="outputs/training_metrics.csv",
        output_dir="outputs"
    )
    print(f"\nTraining plots saved to outputs/")
    
    # Classification report
    save_classification_report(val_labels, val_preds)
        
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