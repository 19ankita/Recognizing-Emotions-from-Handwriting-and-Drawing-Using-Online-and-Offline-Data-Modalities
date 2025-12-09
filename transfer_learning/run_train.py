import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os
import math

from src.dataset import get_dataloaders
from src.model import build_resnet18
from src.utils import accuracy, save_checkpoint


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
def run_train(config_path):
    # Load config file
    cfg = yaml.safe_load(open(config_path, "r"))

    # Prepare output directory
    os.makedirs("outputs", exist_ok=True)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device → {device}")

    # Load data
    train_loader, val_loader, num_classes = get_dataloaders(cfg)

    # Build model
    model = build_resnet18(
        num_classes=num_classes,
        freeze_backbone=cfg["freeze_backbone"]
    ).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (Phase 1: classifier-only or full model depending on freeze_backbone)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=1e-4
    )

    # Scheduler for Phase 1
    scheduler = get_scheduler(
        optimizer,
        warmup_epochs=2,
        total_epochs=cfg["epochs"]
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training history storage
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    best_acc = 0

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(cfg["epochs"]):
        print(f"\n==== Epoch {epoch+1}/{cfg['epochs']} ====")

        # --------------------------------------------------------
        # PHASE SWITCH: Unfreeze Backbone at Epoch 10
        # --------------------------------------------------------
        if epoch == 10:
            print("\n>>> Unfreezing backbone for fine-tuning...")

            # Unfreeze ALL parameters
            model.requires_grad_(True)

            # Rebuild optimizer for full fine-tuning
            optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-5,              # very low LR for fine-tuning
                weight_decay=1e-4
            )

            # Rebuild scheduler for the remaining epochs
            scheduler = get_scheduler(
                optimizer,
                warmup_epochs=0,
                total_epochs=cfg["epochs"] - epoch
            )

        # --------------------------------------------------------
        # TRAINING PHASE
        # --------------------------------------------------------
        model.train()
        train_loss_total = 0
        train_acc_total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images)
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
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)

                with torch.amp.autocast("cuda"):
                    outputs = model(images)
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
            save_checkpoint(model, "outputs/best_model.pth")
            print("Saved new BEST model")

    # Save history file
    with open("outputs/history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining complete! History saved to outputs/history.json")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    run_train(args.config)
