import matplotlib.pyplot as plt
import json
import numpy as np
import argparse

plt.style.use("seaborn-v0_8-whitegrid")


def smooth(values, smoothing=0.6):
    """Exponential moving average smoothing."""
    if len(values) < 2:
        return values
    smoothed = []
    last = values[0]
    for v in values:
        last = last * smoothing + (1 - smoothing) * v
        smoothed.append(last)
    return smoothed


def plot_metrics(history_file,
                 output_path="outputs/training_plot.png",
                 smoothing=0.6):

    with open(history_file, "r") as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    epochs = np.arange(1, len(train_loss) + 1)

    # Smooth curves
    train_loss_s = smooth(train_loss, smoothing)
    val_loss_s = smooth(val_loss, smoothing)
    train_acc_s = smooth(train_acc, smoothing)
    val_acc_s = smooth(val_acc, smoothing)

    plt.figure(figsize=(14, 5), dpi=150)

    # ---------------- LOSS ----------------
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_s, label="Train Loss", linewidth=2.5)
    plt.plot(epochs, val_loss_s, label="Validation Loss", linewidth=2.5)
    plt.scatter(epochs, val_loss, s=14, alpha=0.35)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.title("Loss Curves (Depression Severity Classification)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.25)

    # ---------------- ACCURACY ----------------
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_s, label="Train Accuracy", linewidth=2.5)
    plt.plot(epochs, val_acc_s, label="Validation Accuracy", linewidth=2.5)
    plt.scatter(epochs, val_acc, s=14, alpha=0.35)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy Curves (Depression Severity Classification)", fontsize=14)

    # IMPORTANT: fixed accuracy scale
    plt.ylim(0.0, 1.0)

    plt.legend(fontsize=10)
    plt.grid(alpha=0.25)

    plt.tight_layout()

    # Save outputs
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot → {output_path}")

    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved high-quality PDF → {pdf_path}")

    plt.close()
