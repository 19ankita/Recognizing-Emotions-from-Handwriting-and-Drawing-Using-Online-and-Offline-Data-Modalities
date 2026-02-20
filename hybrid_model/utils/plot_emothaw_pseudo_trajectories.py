import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


def visualize_pseudo_trajectories(
    task="cursive_writing",
    pseudo_root="data/processed/EMOTHAW/pseudo_trajectories",
    image_root="data/raw/EMOTHAW",
    output_root="data/processed/EMOTHAW/visualizations",
    num_samples=6,
    seed=42
):
    random.seed(seed)

    pseudo_dir = os.path.join(pseudo_root, task)
    image_dir  = os.path.join(image_root, task)
    output_dir = os.path.join(output_root, task)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(pseudo_dir):
        print(f"Directory not found: {pseudo_dir}")
        return

    files = [f for f in os.listdir(pseudo_dir) if f.endswith(".npy")]
    if len(files) == 0:
        print("No trajectories found.")
        return

    num_samples = min(num_samples, len(files))
    sampled_files = random.sample(files, num_samples)
    print(f"[{task}] Randomly selected {num_samples} trajectories")

    # --------------------------------------------------
    # SIDE-BY-SIDE (original image | predicted trajectory)
    # --------------------------------------------------
    plt.figure(figsize=(10, 4 * num_samples))

    for i, fname in enumerate(sampled_files):
        traj_path = os.path.join(pseudo_dir, fname)
        traj = np.load(traj_path)

        # assuming the EMOTHAW image has same base name as npy
        image_name = fname.replace(".npy", ".png")
        image_path = os.path.join(image_dir, image_name)

        # Original image
        plt.subplot(num_samples, 2, 2 * i + 1)
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("L")
            plt.imshow(img, cmap="gray")
            plt.title(f"Original: {image_name[:25]}", fontsize=9)
        else:
            plt.text(0.5, 0.5, f"Missing image:\n{image_name}", ha="center", va="center")
            plt.title("Original (missing)", fontsize=9)
        plt.axis("off")

        # Predicted trajectory
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.plot(traj[:, 0], -traj[:, 1], linewidth=1)
        plt.title("Predicted (Pseudo) Trajectory", fontsize=9)
        plt.axis("equal")
        plt.axis("off")

    out_path = os.path.join(output_dir, f"{task}_side_by_side_{num_samples}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved side-by-side comparison to {out_path}")