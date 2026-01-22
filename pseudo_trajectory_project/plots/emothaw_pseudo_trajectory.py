import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# CONFIG
# =========================
TASKS = ["cdt", "cursive_writing", "house", "pentagon", "words"]

TRAJ_ROOT = "data/EMOTHAW/pseudo_trajectories"
IMG_ROOT  = "data/EMOTHAW"
OUT_ROOT  = "figures/pseudo_trajectories"

N_BATCH = 6
FIG_DPI = 300


# =========================
# UTILS
# =========================
def load_traj(path):
    traj = np.load(path)
    return traj[:, 0], traj[:, 1]


def normalize_to_image(x, y, w, h):
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-8) * w
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-8) * h
    return x_n, y_n


# =========================
# MAIN LOOP
# =========================
for task in TASKS:
    print(f"[INFO] Processing task: {task}")

    traj_dir = os.path.join(TRAJ_ROOT, task)
    img_dir  = os.path.join(IMG_ROOT, task)
    out_dir  = os.path.join(OUT_ROOT, task)

    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(traj_dir) if f.endswith(".npy")])
    if len(files) == 0:
        print(f"[WARNING] No trajectories found for {task}")
        continue

    # ---------------------------------
    # 1. SINGLE TRAJECTORY
    # ---------------------------------
    traj_path = os.path.join(traj_dir, files[0])
    x, y = load_traj(traj_path)

    plt.figure(figsize=(4, 4))
    plt.plot(x, y, linewidth=1)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(f"{task}: pseudo-trajectory")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "single_traj.png"), dpi=FIG_DPI)
    plt.close()

    # ---------------------------------
    # 2. TIME-COLORED TRAJECTORY
    # ---------------------------------
    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, c=np.arange(len(x)), cmap="viridis", s=6)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.colorbar(label="Time step")
    plt.title(f"{task}: time-colored trajectory")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "time_colored.png"), dpi=FIG_DPI)
    plt.close()

    # ---------------------------------
    # 3. IMAGE OVERLAY
    # ---------------------------------
    img_name = files[0].replace(".npy", ".png")
    img_path = os.path.join(img_dir, img_name)

    if os.path.exists(img_path):
        img = Image.open(img_path).convert("L")
        x_n, y_n = normalize_to_image(x, y, *img.size)

        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap="gray")
        plt.plot(x_n, y_n, color="red", linewidth=1)
        plt.axis("off")
        plt.title(f"{task}: image overlay")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "overlay.png"), dpi=FIG_DPI)
        plt.close()
    else:
        print(f"[WARNING] Image not found for overlay: {img_name}")

    # ---------------------------------
    # 4. BATCH GRID
    # ---------------------------------
    plt.figure(figsize=(9, 5))

    for i, fname in enumerate(files[:N_BATCH]):
        x, y = load_traj(os.path.join(traj_dir, fname))

        plt.subplot(2, 3, i + 1)
        plt.plot(x, y, linewidth=1)
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.axis("off")
        plt.title(fname[:12])

    plt.suptitle(f"{task}: pseudo-trajectory samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "batch_grid.png"), dpi=FIG_DPI)
    plt.close()

print("All plots generated and saved.")
