import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets.iam_dataset import IAMDataset
from models.build_reverse_model import ReverseModel


def run_sanity_check(
    metadata_path="data/processed/IAM_OnDB/metadata.csv",
    model_path="reverse_model.pth",
    output_path="results/sanity_check_iam.png",
    sample_index=0,
    device=None
):
    """
    Run sanity check by comparing ground truth and predicted trajectory
    for one IAM sample and save the plot.
    """

    # ---------------------------
    # DEVICE
    # ---------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------
    # LOAD DATASET
    # ---------------------------
    dataset = IAMDataset(metadata_path)

    # ---------------------------
    # LOAD MODEL
    # ---------------------------
    model = ReverseModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ---------------------------
    # GET SAMPLE
    # ---------------------------
    img, traj_gt = dataset[sample_index]

    with torch.no_grad():
        traj_pred = model(img.unsqueeze(0).to(device))
        traj_pred = traj_pred.squeeze(0).cpu().numpy()

    traj_gt = traj_gt.numpy()

    # ---------------------------
    # PLOT
    # ---------------------------
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.plot(traj_gt[:, 0], -traj_gt[:, 1])
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    plt.title("Predicted")
    plt.plot(traj_pred[:, 0], -traj_pred[:, 1])
    plt.axis("equal")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Sanity check saved to {output_path}")
