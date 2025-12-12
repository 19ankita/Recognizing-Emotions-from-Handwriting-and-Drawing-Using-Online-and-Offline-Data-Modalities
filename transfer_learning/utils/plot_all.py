import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.dataset import get_dataloaders
from src.model import build_resnet18

from .utils import get_class_names_from_task
from .utils import get_predictions


# ----------------------------------------------------
# PER-CLASS ACCURACY
# ----------------------------------------------------
def plot_class_accuracy(task, task_dir, model_path,
                    img_size=224, batch_size=32, val_ratio=0.2,
                    output="outputs/class_accuracy.pdf"):

    _, val_loader, num_classes = get_dataloaders(
        task=task,
        task_root=task_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=2,
        val_ratio=val_ratio
    )

    class_names = get_class_names_from_task(task_dir, task)

    # Load model
    model = build_resnet18(num_classes, freeze_backbone=False)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    labels, preds = get_predictions(model, val_loader, device)

    # Per-class accuracy
    class_acc = []
    for cls in range(num_classes):
        idx = labels == cls
        acc = (preds[idx] == labels[idx]).mean() if idx.sum() > 0 else 0
        class_acc.append(acc)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_acc, color="steelblue")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=300)
    print(f"Saved → {output}")


# ----------------------------------------------------
# CONFUSION MATRIX
# ----------------------------------------------------
def plot_confmat(task, task_dir, model_path,
            img_size=224, batch_size=32, val_ratio=0.2,
            output="outputs/confusion_matrix.pdf"):

    _, val_loader, num_classes = get_dataloaders(
        task=task,
        task_root=task_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=2,
        val_ratio=val_ratio
    )

    class_names = get_class_names_from_task(task_dir, task)

    model = build_resnet18(num_classes, freeze_backbone=False)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    labels, preds = get_predictions(model, val_loader, device)
    cm = confusion_matrix(labels, preds, normalize="true")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                fmt=".2f")
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=300)
    print(f"Saved → {output}")


# ----------------------------------------------------
# LEARNING CURVE + EARLY STOPPING
# ----------------------------------------------------
def plot_early_stopping(history_path, output="outputs/early_stopping.pdf"):
    history = json.load(open(history_path, "r"))
    val_loss = history["val_loss"]

    best_epoch = val_loss.index(min(val_loss)) + 1
    epochs = range(1, len(val_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch: {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Early Stopping Visualization")
    plt.legend()
    plt.grid()
    plt.savefig(output, dpi=300)
    print(f"Saved → {output}")


def plot_lr(history_path, output="outputs/lr_curve.pdf"):
    history = json.load(open(history_path, "r"))
    lr = history["lr"]

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(lr)+1), lr)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid()
    plt.savefig(output, dpi=300)
    print(f"Saved → {output}")

def run_all_plots(
    task,
    task_dir,
    model_path,
    history_path,
    output_dir="outputs"
):
    plot_class_accuracy(
        task=task,
        task_dir=task_dir,
        model_path=model_path,
        output=os.path.join(output_dir, "class_accuracy.pdf")
    )

    plot_confmat(
        task=task,
        task_dir=task_dir,
        model_path=model_path,
        output=os.path.join(output_dir, "confusion_matrix.pdf")
    )

    plot_early_stopping(
        history_path=history_path,
        output=os.path.join(output_dir, "early_stopping.pdf")
    )

    plot_lr(
        history_path=history_path,
        output=os.path.join(output_dir, "lr_curve.pdf")
    )

    print("\nAll plots saved in outputs/ folder.\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output_dir", default="outputs")

    args = parser.parse_args()

    run_all_plots(
        task=args.task,
        task_dir=args.task_dir,
        model_path=args.model_path,
        history_path=args.history,
        output_dir=args.output_dir
    )
