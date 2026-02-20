import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


def _plot_with_pen(ax, traj, title="Predicted", pen_threshold=0.5):
    """
    traj: [T,3] = (x,y,pen_prob or pen_bin)
    Breaks strokes where pen < threshold.
    """
    xy = traj[:, :2]
    pen = traj[:, 2]

    x = xy[:, 0]
    y = 1.0 - xy[:, 1]  # flip in normalized space (better than -y)

    # Convert pen to 0/1 breaks
    pen_bin = (pen >= pen_threshold).astype(np.int32)

    ax.set_title(title, fontsize=9)
    ax.axis("equal")
    ax.axis("off")

    start = 0
    for i in range(1, len(x)):
        if pen_bin[i] == 0:
            seg_x = x[start:i]
            seg_y = y[start:i]
            if len(seg_x) > 1:
                ax.plot(seg_x, seg_y, linewidth=1)
            start = i

    seg_x = x[start:]
    seg_y = y[start:]
    if len(seg_x) > 1:
        ax.plot(seg_x, seg_y, linewidth=1)


def visualize_pseudo_trajectories(
    task="cursive_writing",
    pseudo_root="data/processed/EMOTHAW/pseudo_trajectories",
    image_root="data/raw/EMOTHAW",
    output_root="data/processed/EMOTHAW/visualizations",
    num_samples=2,
    seed=42,
    pen_threshold=0.5
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

    plt.figure(figsize=(10, 4 * num_samples))

    for i, fname in enumerate(sampled_files):
        traj_path = os.path.join(pseudo_dir, fname)
        traj = np.load(traj_path)

        # match image name
        base = os.path.splitext(fname)[0]
        # try png then jpg just in case
        image_path_png = os.path.join(image_dir, base + ".png")
        image_path_jpg = os.path.join(image_dir, base + ".jpg")
        image_path = image_path_png if os.path.exists(image_path_png) else image_path_jpg

        # Original image
        ax1 = plt.subplot(num_samples, 2, 2 * i + 1)
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("L")
            ax1.imshow(img, cmap="gray")
            ax1.set_title(f"Original: {os.path.basename(image_path)[:25]}", fontsize=9)
        else:
            ax1.text(0.5, 0.5, f"Missing image:\n{base}", ha="center", va="center")
            ax1.set_title("Original (missing)", fontsize=9)
        ax1.axis("off")

        # Predicted trajectory (with pen breaks)
        ax2 = plt.subplot(num_samples, 2, 2 * i + 2)
        if traj.shape[1] >= 3:
            _plot_with_pen(ax2, traj, title="Predicted (Pseudo) Trajectory", pen_threshold=pen_threshold)
        else:
            # fallback if old [T,2]
            ax2.plot(traj[:, 0], 1.0 - traj[:, 1], linewidth=1)
            ax2.set_title("Predicted (no pen)", fontsize=9)
            ax2.axis("equal")
            ax2.axis("off")

    out_path = os.path.join(output_dir, f"{task}_side_by_side_{num_samples}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved side-by-side comparison to {out_path}")