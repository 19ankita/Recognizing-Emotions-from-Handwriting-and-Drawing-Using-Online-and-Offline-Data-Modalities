import os
import pandas as pd
import matplotlib.pyplot as plt

NO_AUG_ROOT = "results_no_aug"
AUG_ROOT = "results_aug"

TASKS = ["pentagon", "house", "cdt", "cursive_writing", "words"]


def load_metrics(result_root, task):
    task_dir = os.path.join(result_root, task)
    if not os.path.isdir(task_dir):
        return None

    timestamps = sorted(os.listdir(task_dir))
    if len(timestamps) == 0:
        return None

    last_run = os.path.join(task_dir, timestamps[-1])
    csv_path = os.path.join(last_run, "metrics.csv")

    if not os.path.isfile(csv_path):
        return None

    df = pd.read_csv(csv_path)
    return df


def plot_task(task):

    df_no_aug = load_metrics(NO_AUG_ROOT, task)
    df_aug = load_metrics(AUG_ROOT, task)

    if df_no_aug is None or df_aug is None:
        print(f"Skipping {task}: missing files.")
        return

    # Filter split
    no_train = df_no_aug[df_no_aug["split"] == "train"]
    no_val = df_no_aug[df_no_aug["split"] == "val"]

    aug_train = df_aug[df_aug["split"] == "train"]
    aug_val = df_aug[df_aug["split"] == "val"]

    # ----------------- Plot -----------------
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(no_train["epoch"], no_train["loss"], label="Train Loss (No Aug)", color="blue")
    plt.plot(no_val["epoch"], no_val["loss"], label="Val Loss (No Aug)", color="cyan")
    plt.plot(aug_train["epoch"], aug_train["loss"], label="Train Loss (Aug)", color="red")
    plt.plot(aug_val["epoch"], aug_val["loss"], label="Val Loss (Aug)", color="orange")
    plt.title(f"Loss – {task}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(no_train["epoch"], no_train["acc"], label="Train Acc (No Aug)", color="blue")
    plt.plot(no_val["epoch"], no_val["acc"], label="Val Acc (No Aug)", color="cyan")
    plt.plot(aug_train["epoch"], aug_train["acc"], label="Train Acc (Aug)", color="red")
    plt.plot(aug_val["epoch"], aug_val["acc"], label="Val Acc (Aug)", color="orange")
    plt.title(f"Accuracy – {task}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Save
    out_path = f"{task}_aug_vs_no_aug.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("\n=== Generating augmentation comparison plots ===\n")

    for task in TASKS:
        plot_task(task)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
