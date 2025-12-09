import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os

from src.dataset import get_dataloaders
from model import build_resnet18
from utils import accuracy, save_checkpoint


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

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"]
    )

    # Store training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_acc = 0

    # Training loop
    for epoch in range(cfg["epochs"]):
        print(f"\n==== Epoch {epoch+1}/{cfg['epochs']} ====")

        # -----------------
        # Training phase
        # -----------------
        model.train()
        train_loss_total = 0
        train_acc_total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_acc_total += accuracy(outputs, labels)

        train_loss = train_loss_total / len(train_loader)
        train_acc = train_acc_total / len(train_loader)

        # -----------------
        # Validation phase
        # -----------------
        model.eval()
        val_loss_total = 0
        val_acc_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item()
                val_acc_total += accuracy(outputs, labels)

        val_loss = val_loss_total / len(val_loader)
        val_acc = val_acc_total / len(val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, "outputs/best_model.pth")

    # Save training curves
    with open("outputs/history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("Training complete! History saved to outputs/history.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    run_train(args.config)
