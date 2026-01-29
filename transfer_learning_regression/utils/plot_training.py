import matplotlib.pyplot as plt
import json
import numpy as np

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
    """
    Regression training curves:
    - Train MSE
    - Validation MSE
    """

    with open(history_file, "r") as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    epochs = np.arange(1, len(train_loss) + 1)

    # Smooth curves
    train_loss_s = smooth(train_loss, smoothing)
    val_loss_s = smooth(val_loss, smoothing)

    # Figure
    plt.figure(figsize=(8, 5), dpi=120)

    # ---------------- LOSS PLOT ----------------
    plt.plot(epochs, train_loss_s, label="Train MSE", linewidth=2.5)
    plt.plot(epochs, val_loss_s, label="Validation MSE", linewidth=2.5)
    plt.scatter(epochs, val_loss, s=15, alpha=0.4)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Squared Error", fontsize=12)
    plt.title("Training vs Validation Loss (Regression)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    # Save PNG
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot → {output_path}")

    # Save PDF (thesis quality)
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved high-quality PDF → {pdf_path}")

    plt.close()
