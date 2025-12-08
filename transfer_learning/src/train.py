import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from utils import accuracy, save_checkpoint
from dataset import get_dataloaders
from model import build_resnet18


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0, 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(outputs, labels)

    return total_loss / len(loader), total_acc / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)

    return total_loss / len(loader), total_acc / len(loader)


def main(config_path):
    cfg = yaml.safe_load(open(config_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, num_classes = get_dataloaders(cfg)
    model = build_resnet18(num_classes, freeze_backbone=cfg["freeze_backbone"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    best_acc = 0
    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, "outputs/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
