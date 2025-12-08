import matplotlib.pyplot as plt
import json
import argparse


def plot_metrics(history_file, output_path="outputs/training_plot.png"):
    with open(history_file, "r") as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    args = parser.parse_args()

    plot_metrics(args.history)
