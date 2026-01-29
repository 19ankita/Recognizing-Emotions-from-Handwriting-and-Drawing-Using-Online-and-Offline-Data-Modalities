import matplotlib.pyplot as plt
import json
import numpy as np
import argparse

plt.style.use("seaborn-v0_8-whitegrid")  # clean scientific style


def smooth(values, smoothing=0.6):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        last = last * smoothing + (1 - smoothing) * v
        smoothed.append(last)
    return smoothed


def plot_metrics(history_file, output_path="outputs/training_plot.png", smoothing=0.6):
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

    # Figure
    plt.figure(figsize=(14, 5), dpi=120)

    # ---------------- LOSS PLOT ----------------
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_s, label="Train Loss", linewidth=2.5)
    plt.plot(epochs, val_loss_s, label="Val Loss", linewidth=2.5)
    plt.scatter(epochs, val_loss, s=10, alpha=0.4)  # raw points for reference
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training vs Validation Loss", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # ---------------- ACCURACY PLOT ----------------
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_s, label="Train Accuracy", linewidth=2.5)
    plt.plot(epochs, val_acc_s, label="Val Accuracy", linewidth=2.5)
    plt.scatter(epochs, val_acc, s=10, alpha=0.4)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Training vs Validation Accuracy", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    # Save PNG
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot → {output_path}")

    # Save PDF (high quality for thesis)
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved high-quality PDF → {pdf_path}")

