import json
import os
import matplotlib.pyplot as plt


# ----------------------------------------------------
# EARLY STOPPING (VALIDATION LOSS)
# ----------------------------------------------------
def plot_early_stopping(history_path, output="outputs/early_stopping.pdf"):
    history = json.load(open(history_path, "r"))
    val_loss = history["val_loss"]

    best_epoch = val_loss.index(min(val_loss)) + 1
    epochs = range(1, len(val_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.axvline(best_epoch, color="red", linestyle="--",
                label=f"Best Epoch: {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Early Stopping (Regression)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Saved → {output}")


# ----------------------------------------------------
# LEARNING RATE CURVE
# ----------------------------------------------------
def plot_lr(history_path, output="outputs/lr_curve.pdf"):
    history = json.load(open(history_path, "r"))
    lr = history["lr"]

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(lr) + 1), lr)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Saved → {output}")


# ----------------------------------------------------
# RUN ALL PLOTS
# ----------------------------------------------------
def run_all_plots(
    task,
    task_dir,
    model_path,
    history_path,
    output_dir="outputs"
):
    os.makedirs(output_dir, exist_ok=True)

    plot_early_stopping(
        history_path=history_path,
        output=os.path.join(output_dir, "early_stopping.pdf")
    )

    plot_lr(
        history_path=history_path,
        output=os.path.join(output_dir, "lr_curve.pdf")
    )

    print("\nRegression plots saved in outputs/ folder.\n")
