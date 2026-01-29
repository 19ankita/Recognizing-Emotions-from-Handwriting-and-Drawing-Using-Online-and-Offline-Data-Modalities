import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from ..src.dataset import get_dataloaders
from ..src.model import build_resnet18

import inspect
print(inspect.getfile(get_dataloaders))

DASS_LABELS = ["Depression", "Anxiety", "Stress", "Total"]


# ------------------------------------------------------------
# COLLECT PREDICTIONS
# ------------------------------------------------------------
def collect_predictions(model, dataloader, device):
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for images, pseudo, labels in dataloader:
            images = images.to(device)
            pseudo = pseudo.to(device)

            preds = model(images, pseudo)

            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    return np.vstack(y_true), np.vstack(y_pred)


# ------------------------------------------------------------
# METRIC COMPUTATION
# ------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    results = {}

    for i, name in enumerate(DASS_LABELS):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r, p = pearsonr(y_true[:, i], y_pred[:, i])

        results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "Pearson_r": r,
            "Pearson_p": p
        }

    return results


# ------------------------------------------------------------
# SCATTER PLOTS
# ------------------------------------------------------------
def plot_scatter(y_true, y_pred, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, name in enumerate(DASS_LABELS):
        plt.figure(figsize=(5, 5), dpi=120)

        plt.scatter(
            y_true[:, i],
            y_pred[:, i],
            alpha=0.6,
            edgecolor="k"
        )

        min_v = min(y_true[:, i].min(), y_pred[:, i].min())
        max_v = max(y_true[:, i].max(), y_pred[:, i].max())

        plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2)

        r, _ = pearsonr(y_true[:, i], y_pred[:, i])

        plt.xlabel("Ground Truth")
        plt.ylabel("Prediction")
        plt.title(f"{name} (r = {r:.2f})")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        png_path = os.path.join(output_dir, f"{name.lower()}_scatter.png")
        pdf_path = png_path.replace(".png", ".pdf")

        plt.savefig(png_path)
        plt.savefig(pdf_path)
        plt.close()

        print(f"Saved → {png_path}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device → {device}")

    # Load validation data only
    _, val_loader, _ = get_dataloaders(
        task=args.task,
        task_root=args.task_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        label_csv=args.label_csv
    )

    # Load model
    model = build_resnet18(
        output_dim=4,
        freeze_backbone=False
    )

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Collect predictions
    y_true, y_pred = collect_predictions(model, val_loader, device)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)

    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "regression_metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nPer-dimension regression metrics:")
    for k, v in metrics.items():
        print(f"{k}: RMSE={v['RMSE']:.3f}, MAE={v['MAE']:.3f}, r={v['Pearson_r']:.3f}")

    print(f"\nSaved metrics → {metrics_path}")

    # Scatter plots
    plot_scatter(
        y_true,
        y_pred,
        output_dir=os.path.join(args.output_dir, "scatter_plots")
    )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--task_dir", type=str, required=True)
    parser.add_argument("--label_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--output_dir", type=str, default="outputs/regression_eval")

    args = parser.parse_args()
    main(args)
