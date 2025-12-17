import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import umap

from src.dataset import get_dataloaders
from src.model import build_resnet18, build_resnet50
from utils.helper import get_class_names_from_task


# ------------------------------------------------------------
# Extract backbone embeddings
# ------------------------------------------------------------
def extract_embeddings(model, dataloader, device):
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for images, pseudo, y in dataloader:
            images = images.to(device)

            # backbone features only
            feats = model.backbone(images)

            embeddings.append(feats.cpu())
            labels.append(y)

    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()


# ------------------------------------------------------------
# Plot embeddings
# ------------------------------------------------------------
def plot_embeddings(X, y, class_names, method, save_path):
    plt.figure(figsize=(7, 6))

    for cls_idx, cls_name in enumerate(class_names):
        idxs = y == cls_idx
        plt.scatter(
            X[idxs, 0],
            X[idxs, 1],
            label=cls_name,
            alpha=0.6,
            s=20
        )

    plt.legend()
    plt.title(f"{method.upper()} of CNN embeddings")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def plot_stress_binary_tsne(X_2d, y, class_names, save_path):
    """
    Stress vs Non-Stress visualization
    """

    stress_idx = class_names.index("stress")

    binary_labels = (y == stress_idx).astype(int)

    plt.figure(figsize=(7, 6))

    plt.scatter(
        X_2d[binary_labels == 0, 0],
        X_2d[binary_labels == 0, 1],
        label="Non-Stress",
        alpha=0.4,
        s=20
    )

    plt.scatter(
        X_2d[binary_labels == 1, 0],
        X_2d[binary_labels == 1, 1],
        label="Stress",
        alpha=0.6,
        s=30
    )

    plt.legend()
    plt.title("Stress vs Non-Stress (t-SNE)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data (VAL only)
    _, val_loader, num_classes = get_dataloaders(
        task=args.task,
        task_root=args.task_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=0.2
    )

    class_names = get_class_names_from_task(args.task_dir, args.task)

    # Build model
    if args.model == "resnet18":
        model = build_resnet18(num_classes, freeze_backbone=True)
    else:
        model = build_resnet50(num_classes, freeze_backbone=True)

    model = model.to(device)

    # Load checkpoint
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device)
    )

    # Extract embeddings
    X, y = extract_embeddings(model, val_loader, device)

    # Dimensionality reduction
    if args.method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)

    X_2d = reducer.fit_transform(X)

    # Save plot
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(
        args.output_dir,
        f"{args.method}_{args.task}_{args.model}.pdf"
    )

    plot_embeddings(X_2d, y, class_names, args.method, save_path)
    
    # --------------------------------------------------
    # Stress vs Non-Stress t-SNE
    # --------------------------------------------------
    save_path_binary = os.path.join(
        args.output_dir,
        f"{args.method}_{args.task}_{args.model}_stress_binary.pdf"
    )
    plot_stress_binary_tsne(X_2d, y, class_names, save_path_binary)
    
    print("Saved embedding visualizations:")
    print(" -", save_path)
    print(" -", save_path_binary)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--task_dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet50"])
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--method", type=str, choices=["tsne", "umap"], default="tsne")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--output_dir", type=str, default="outputs/embeddings")

    args = parser.parse_args()
    main(args)
