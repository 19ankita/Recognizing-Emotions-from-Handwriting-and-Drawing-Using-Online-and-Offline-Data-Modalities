from src.datasets.iam_dataset import IAMDataset
import matplotlib.pyplot as plt
import os

# ---------------------------
# CONFIG
# ---------------------------
METADATA = "data/IAM_OnDB/processed/metadata.csv"
OUT_DIR = "data/IAM_OnDB/processed/overlays"
NUM_SAMPLES = 5

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# LOAD DATASET
# ---------------------------
dataset = IAMDataset(metadata_csv=METADATA)

img, traj = dataset[0]
print("Image shape:", img.shape)
print("Trajectory shape:", traj.shape)

# ---------------------------
# SAVE OVERLAY PLOTS
# ---------------------------
for i in range(NUM_SAMPLES):
    img, traj = dataset[i]

    h, w = img.shape[-2], img.shape[-1]

    plt.figure(figsize=(4, 4))
    plt.imshow(img.squeeze(), cmap="gray")

    plt.plot(
        traj[:, 0] * w,
        traj[:, 1] * h,
        color="red",
        linewidth=1
    )

    plt.axis("off")
    plt.title(f"Sample {i}")

    out_path = os.path.join(OUT_DIR, f"overlay_{i}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")
