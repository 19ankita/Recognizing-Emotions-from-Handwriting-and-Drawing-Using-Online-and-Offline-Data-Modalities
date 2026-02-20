import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------
# CONFIG
# ---------------------------
EMOTHAW_IMG_ROOT = "data/EMOTHAW"
PSEUDO_TRAJ_ROOT = "data/EMOTHAW/pseudo_trajectories"
OUT_ROOT = "plots/emothaw_overlays"

TASKS = [
    "cdt",
    "cursive_writing",
    "house",
    "pentagon",
    "words"
]

NUM_SAMPLES_PER_TASK = 5  # adjust if needed

os.makedirs(OUT_ROOT, exist_ok=True)

# ---------------------------
# OVERLAY GENERATION
# ---------------------------
for task in TASKS:
    img_dir = os.path.join(EMOTHAW_IMG_ROOT, task)
    traj_dir = os.path.join(PSEUDO_TRAJ_ROOT, task)
    out_dir = os.path.join(OUT_ROOT, task)

    os.makedirs(out_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])[:NUM_SAMPLES_PER_TASK]

    print(f"[{task}] Generating {len(image_files)} overlays")

    for fname in image_files:
        img_path = os.path.join(img_dir, fname)
        traj_path = os.path.join(
            traj_dir,
            fname.rsplit(".", 1)[0] + ".npy"
        )

        if not os.path.exists(traj_path):
            print(f"  Skipping {fname} (no trajectory)")
            continue

        # Load image
        img = Image.open(img_path).convert("L")
        img_np = np.array(img)

        # Load trajectory
        traj = np.load(traj_path)

        h, w = img_np.shape

        x = traj[:, 0] * w
        y = traj[:, 1] * h

        # ---------------------------
        # Plot
        # ---------------------------
        plt.figure(figsize=(4, 4))
        plt.imshow(img_np, cmap="gray")

        plt.plot(x, y, color="red", linewidth=1.0)

        plt.axis("off")
        plt.title(task.replace("_", " ").title(), fontsize=11)

        out_path = os.path.join(
            out_dir,
            fname.rsplit(".", 1)[0] + "_overlay.png"
        )

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

print("EMOTHAW overlay visualizations saved")
