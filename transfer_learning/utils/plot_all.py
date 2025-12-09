import torch
import json
import yaml
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from src.eval_utils import get_predictions
from src.dataset import get_dataloaders
from src.model import build_resnet18

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------
# 1. Per-Class Accuracy
# ----------------------------------------------------
def plot_class_accuracy(config_path, model_path, output=None):
    
    if output is None:
        output = os.path.join("outputs", "class_accuracy.png")
    
    cfg = yaml.safe_load(open(config_path, "r"))

    _, val_loader, num_classes = get_dataloaders(cfg)

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
    plt.bar(range(num_classes), class_acc)
    plt.xlabel("Class Index")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(range(num_classes))
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved → {output}")

# ----------------------------------------------------
# 2. Confusion Matrix
# ---------------------------------------------------
def plot_confmat(config_path, model_path, output=None):
      
    if output is None:
       output = os.path.join("outputs", "confusion_matrix.png")

    # Load config (yaml)
    cfg = yaml.safe_load(open(config_path, "r"))

    # Load data
    _, val_loader, num_classes = get_dataloaders(cfg)

    # Load model
    model = build_resnet18(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Predictions
    labels, preds = get_predictions(model, val_loader, device)

    cm = confusion_matrix(labels, preds, normalize="true")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f")
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    
    plt.savefig(output)
    print(f"Saved: {output}")
    
    
# ----------------------------------------------------
# 3. Interactive Dashboard
# ----------------------------------------------------    
def plot_dashboard(history_file, output=None):
    
    if output is None:
        output = os.path.join("outputs", "dashboard.png")
        
    history = json.load(open(history_file, "r"))

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss Curves", "Accuracy Curves")
    )

    # Loss
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"],
                            mode="lines", name="Train Loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],
                            mode="lines", name="Val Loss"), row=1, col=1)

    # Accuracy
    fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"],
                            mode="lines", name="Train Acc"), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"],
                            mode="lines", name="Val Acc"), row=1, col=2)

    fig.update_layout(title="Interactive Training Dashboard", width=1000, height=450)
    fig.write_html(output)
    print(f"Saved interactive dashboard → {output}")


# ----------------------------------------------------
# 4. Early Stopping Visualization
# ----------------------------------------------------
def plot_early_stopping(history_file, output=None):
    
    if output is None:
        output = os.path.join("outputs", "early_stopping.png")
        
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
    plt.savefig(output)
    print(f"Saved → {output}")

# ----------------------------------------------------
# 5. Learning Rate Curve
# ----------------------------------------------------
def plot_lr(history_file, output=None):
    
    if output is None:
        output = os.path.join("outputs", "lr_curve.png")
        
    history = json.load(open(history_file, "r"))
    lr = history["lr"]
    epochs = range(1, len(lr) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lr, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.savefig(output)
    print(f"Saved → {output}")