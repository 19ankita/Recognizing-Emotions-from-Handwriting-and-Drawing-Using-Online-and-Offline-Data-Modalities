import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


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


def plot_metrics_for_state(csv_path, state, output_dir="outputs", smoothing=0.6):

    df = pd.read_csv(csv_path)
    
    epochs = df["epoch"].values
    train_loss = df["train_loss"].values
    val_loss = df["val_loss"].values
    train_acc = df["train_acc"].values
    val_acc = df["val_acc"].values

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

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Loss Curves ({state.capitalize()} Severity Classification)")
    plt.legend()
    plt.grid(alpha=0.25)

# ---------------- ACCURACY ----------------
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_s, label="Train Accuracy", linewidth=2.5)
    plt.plot(epochs, val_acc_s, label="Validation Accuracy", linewidth=2.5)
    plt.scatter(epochs, val_acc, s=14, alpha=0.35)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curves ({state.capitalize()} Severity Classification)")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()

    png_path = os.path.join(output_dir, f"training_curves_{state}.png")
    pdf_path = png_path.replace(".png", ".pdf")

    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to {png_path} and {pdf_path}")
    
def plot_all_states(task, output_dir="outputs"):
    
    """
    Plot training curves for depression, anxiety, and stress.
    Assumes CSVs are named:
    training_metrics_<task>_<state>.csv
    """

    states = ["depression", "anxiety", "stress"]

    for state in states:
        csv_path = os.path.join(
            output_dir,
            f"training_metrics_{task}_{state}.csv"
        )

        if not os.path.exists(csv_path):
            print(f"[WARNING] Missing file: {csv_path}")
            continue

        plot_metrics_for_state(
            csv_path=csv_path,
            state=state,
            output_dir=output_dir
        )
