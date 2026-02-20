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
    seed=42,
    overlay=False,          
    grid=True,              
    grid_cols=3            
):
    random.seed(seed)

    pseudo_dir = os.path.join(pseudo_root, task)
    image_dir = os.path.join(image_root, task)
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

    # ==================================================
    # 1) OVERLAY MODE (all trajectories in one plot)
    # ==================================================
    if overlay:
        plt.figure(figsize=(7, 7))
        for fname in sampled_files:
            traj = np.load(os.path.join(pseudo_dir, fname))
            plt.plot(traj[:, 0], -traj[:, 1], linewidth=1)

        plt.title(f"{task} — Overlay of {num_samples} pseudo trajectories")
        plt.axis("equal")
        plt.axis("off")

        overlay_path = os.path.join(output_dir, f"{task}_overlay_{num_samples}.png")
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Overlay saved to {overlay_path}")
        return

    # ==================================================
    # 2) GRID MODE (multiple trajectories in a grid)
    # ==================================================
    if grid:
        rows = int(np.ceil(num_samples / grid_cols))
        plt.figure(figsize=(4 * grid_cols, 4 * rows))

        for i, fname in enumerate(sampled_files):
            traj = np.load(os.path.join(pseudo_dir, fname))
            plt.subplot(rows, grid_cols, i + 1)
            plt.plot(traj[:, 0], -traj[:, 1], linewidth=1)
            plt.title(fname[:18], fontsize=8)
            plt.axis("equal")
            plt.axis("off")

        grid_path = os.path.join(output_dir, f"{task}_grid_{num_samples}.png")
        plt.tight_layout()
        plt.savefig(grid_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Grid saved to {grid_path}")
        return

    # ==================================================
    # 3) SIDE-BY-SIDE MODE (your original: image + traj)
    # ==================================================
    plt.figure(figsize=(10, 4 * num_samples))

    for i, fname in enumerate(sampled_files):
        traj = np.load(os.path.join(pseudo_dir, fname))

        image_name = fname.replace(".npy", ".png")  # adjust if needed
        image_path = os.path.join(image_dir, image_name)

        # Original image
        plt.subplot(num_samples, 2, 2 * i + 1)
        img = Image.open(image_path).convert("L")
        plt.imshow(img, cmap="gray")
        plt.title(f"Original: {image_name[:15]}", fontsize=8)
        plt.axis("off")

        # Pseudo trajectory
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.plot(traj[:, 0], -traj[:, 1])
        plt.title("Pseudo Trajectory", fontsize=8)
        plt.axis("equal")
        plt.axis("off")

    comparison_path = os.path.join(output_dir, f"{task}_comparison.png")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Side-by-side comparison saved to {comparison_path}")