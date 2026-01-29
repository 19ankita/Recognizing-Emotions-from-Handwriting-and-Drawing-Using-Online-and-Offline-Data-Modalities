import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


def plot_training_from_csv(csv_path, output_dir="outputs"):
    df = pd.read_csv(csv_path)

    epochs = df["epoch"]

    # -------- RMSE --------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["val_rmse"], label="Validation RMSE", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Validation RMSE over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/val_rmse_curve.pdf", dpi=300)
    plt.close()

    # -------- R² --------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["val_r2"], label="Validation R²", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("Validation R² over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/val_r2_curve.pdf", dpi=300)
    plt.close()

    print("Saved RMSE and R² training curves.")
