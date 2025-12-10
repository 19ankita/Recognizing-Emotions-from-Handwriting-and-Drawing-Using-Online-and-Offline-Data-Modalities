import torch
import json
import yaml
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from src.dataset import get_dataloaders
from src.model import build_resnet18

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_predictions(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    return all_labels, all_preds


# ----------------------------------------------------
# 1. Per-Class Accuracy
# ----------------------------------------------------
def plot_class_accuracy(config_path, model_path, output=None):
    
    if output is None:
        output = os.path.join("outputs", "class_accuracy.pdf")
    
    cfg = yaml.safe_load(open(config_path, "r"))

    _, val_loader, num_classes = get_dataloaders(cfg)
    
    # Handles both ImageFolder and Subset
    if hasattr(val_loader.dataset, "classes"):
        class_names = val_loader.dataset.classes
    else:
        class_names = val_loader.dataset.dataset.classes


    model = build_resnet18(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    labels, preds = get_predictions(model, val_loader, device)

    class_acc = []
    for cls in range(num_classes):
        idx = labels == cls
        acc = (preds[idx] == labels[idx]).mean() if idx.sum() > 0 else 0
        class_acc.append(acc)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_acc)
    plt.xlabel("Classes")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=300, format="pdf")
    print(f"Saved → {output}")

# ----------------------------------------------------
# 2. Confusion Matrix
# ---------------------------------------------------
def plot_confmat(config_path, model_path, output=None):
      
    if output is None:
       output = os.path.join("outputs", "confusion_matrix.pdf")

    # Load config (yaml)
    cfg = yaml.safe_load(open(config_path, "r"))

    # Load data
    _, val_loader, num_classes = get_dataloaders(cfg)
    
    # Handles both ImageFolder and Subset
    if hasattr(val_loader.dataset, "classes"):
        class_names = val_loader.dataset.classes
    else:
        class_names = val_loader.dataset.dataset.classes

    # Load model
    model = build_resnet18(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Predictions
    labels, preds = get_predictions(model, val_loader, device)

    cm = confusion_matrix(labels, preds, normalize="true")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True, 
                cmap="Blues", 
                fmt=".2f",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=300, format="pdf")
    print(f"Saved: {output}")
    
  
# ----------------------------------------------------
# 4. Early Stopping Visualization
# ----------------------------------------------------
def plot_early_stopping(history_file, output=None):
    
    if output is None:
        output = os.path.join("outputs", "early_stopping.pdf")
        
    history = json.load(open(history_file, "r"))
    val_loss = history["val_loss"]

    best_epoch = val_loss.index(min(val_loss)) + 1
    epochs = range(1, len(val_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch: {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Early Stopping Visualization")
    plt.legend()
    plt.grid(True)
    plt.savefig(output, dpi=300, format="pdf")
    print(f"Saved → {output}")

# ----------------------------------------------------
# 5. Learning Rate Curve
# ----------------------------------------------------
def plot_lr(history_file, output=None):
    
    if output is None:
        output = os.path.join("outputs", "lr_curve.pdf")
        
    history = json.load(open(history_file, "r"))
    lr = history["lr"]
    epochs = range(1, len(lr) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lr, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.savefig(output, dpi=300, format="pdf")
    print(f"Saved → {output}")
    
if __name__ == "__main__":
    config = "configs/default.yaml"
    model_path = "outputs/best_model.pth"
    history_path = "outputs/history.json"

    plot_class_accuracy(config, model_path)
    plot_confmat(config, model_path)
    plot_early_stopping(history_path)
    plot_lr(history_path)

    print("\nAll plots saved in outputs/ folder.\n")
