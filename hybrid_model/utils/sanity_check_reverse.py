import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from datasets.iam_dataset import IAMDataset
from models.build_reverse_model import ReverseModel


def _plot_with_pen(ax, xy, pen, title, flip_y=True):
    """
    xy:  [T,2] normalized in [0,1]
    pen: [T]   0/1 (GT) or boolean/binary (pred after threshold)
    """
    x = xy[:, 0]
    y = xy[:, 1]
    if flip_y:
        y = 1.0 - y  # better than -y for normalized coords

    ax.set_title(title)
    ax.axis("equal")
    ax.axis("off")

    # break whenever pen == 0 (stroke start / pen-up transition)
    start = 0
    for i in range(1, len(x)):
        if pen[i] == 0:
            seg_x = x[start:i]
            seg_y = y[start:i]
            if len(seg_x) > 1:
                ax.plot(seg_x, seg_y, linewidth=1)
            start = i

    # last segment
    seg_x = x[start:]
    seg_y = y[start:]
    if len(seg_x) > 1:
        ax.plot(seg_x, seg_y, linewidth=1)


def run_sanity_check(
    metadata_path="data/processed/IAM_OnDB/metadata.csv",
    model_path="reverse_model.pth",
    output_path="results/sanity_check_iam.png",
    sample_index=0,
    device=None
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = IAMDataset(metadata_path)

    model = ReverseModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    img, traj_gt = dataset[sample_index]  # expected [T,3] = x,y,pen
    traj_gt_np = traj_gt.numpy()

    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).squeeze(0)  # [T,3]
        pred_np = pred.cpu().numpy()

    # split GT
    gt_xy = traj_gt_np[:, :2]
    gt_pen = (traj_gt_np[:, 2] > 0.5).astype(np.int32)

    # split Pred: pen is logits -> sigmoid -> threshold
    pred_xy = np.clip(pred_np[:, :2], 0.0, 1.0)
    pred_pen_prob = 1.0 / (1.0 + np.exp(-pred_np[:, 2]))
    pred_pen = (pred_pen_prob > 0.5).astype(np.int32)

    print("GT xy range:",
          gt_xy[:, 0].min(), gt_xy[:, 0].max(),
          gt_xy[:, 1].min(), gt_xy[:, 1].max())
    print("GT pen unique:", np.unique(gt_pen))
    print("Pred pen prob min/max:", pred_pen_prob.min(), pred_pen_prob.max())

    plt.figure(figsize=(10, 4))

    ax1 = plt.subplot(1, 2, 1)
    _plot_with_pen(ax1, gt_xy, gt_pen, "Ground Truth (pen breaks)", flip_y=True)

    ax2 = plt.subplot(1, 2, 2)
    _plot_with_pen(ax2, pred_xy, pred_pen, "Predicted (pen breaks)", flip_y=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Sanity check saved to {output_path}")


if __name__ == "__main__":
    run_sanity_check()