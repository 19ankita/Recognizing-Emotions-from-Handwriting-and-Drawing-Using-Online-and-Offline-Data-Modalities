import os
import numpy as np
import matplotlib.pyplot as plt
import random


def visualize_pseudo_trajectories(
    task="cursive_writing",
    pseudo_root="data/processed/EMOTHAW/pseudo_trajectories",
    output_root="data/processed/EMOTHAW/visualizations",
    num_samples=9,
    overlay=True,
    seed=42
):
    """
    Advanced visualization of pseudo-trajectories.

    Features:
    - Random sampling
    - Grid plotting
    - Overlay mode (all trajectories in one plot)
    - Automatically saves figures
    """

    random.seed(seed)

    pseudo_dir = os.path.join(pseudo_root, task)
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
    # GRID PLOT
    # --------------------------------------------------
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    plt.figure(figsize=(12, 12))

    for i, fname in enumerate(sampled_files):
        traj = np.load(os.path.join(pseudo_dir, fname))

        plt.subplot(grid_size, grid_size, i + 1)
        plt.plot(traj[:, 0], -traj[:, 1])
        plt.title(fname[:15], fontsize=8)
        plt.axis("equal")
        plt.axis("off")

    grid_path = os.path.join(output_dir, f"{task}_grid.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Grid saved to {grid_path}")

    # --------------------------------------------------
    # OVERLAY PLOT
    # --------------------------------------------------
    if overlay:
        plt.figure(figsize=(8, 8))

        for fname in sampled_files:
            traj = np.load(os.path.join(pseudo_dir, fname))
            plt.plot(traj[:, 0], -traj[:, 1], alpha=0.7)

        plt.title(f"{task} - Overlay Comparison")
        plt.axis("equal")

        overlay_path = os.path.join(output_dir, f"{task}_overlay.png")
        plt.savefig(overlay_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Overlay saved to {overlay_path}")
