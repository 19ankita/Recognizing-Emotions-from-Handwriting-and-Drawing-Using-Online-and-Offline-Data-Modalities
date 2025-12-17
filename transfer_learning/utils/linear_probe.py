import argparse
import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import get_dataloaders
from src.model import build_resnet18, build_resnet50
from utils.helper import get_class_names_from_task


# ------------------------------------------------------------
# Extract frozen embeddings
# ------------------------------------------------------------
def extract_embeddings(model, dataloader, device):
    model.eval()

    feats = []
    labels = []

    with torch.no_grad():
        for images, pseudo, y in dataloader:
            images = images.to(device)
            emb = model.backbone(images)
            feats.append(emb.cpu())
            labels.append(y)

    return (
        torch.cat(feats).numpy(),
        torch.cat(labels).numpy()
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data
    train_loader, val_loader, num_classes = get_dataloaders(
        task=args.task,
        task_root=args.task_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=0.2
    )

    class_names = get_class_names_from_task(args.task_dir, args.task)

    # Load frozen CNN
    if args.model == "resnet18":
        model = build_resnet18(num_classes, freeze_backbone=True)
    else:
        model = build_resnet50(num_classes, freeze_backbone=True)

    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device)
    )
    model = model.to(device)

    # Extract embeddings
    X_train, y_train = extract_embeddings(model, train_loader, device)
    X_val, y_val = extract_embeddings(model, val_loader, device)

    print("Embedding shape:", X_train.shape)

    # --------------------------------------------------------
    # Train linear classifier
    # --------------------------------------------------------
    clf = LogisticRegression(
        max_iter=500,
        multi_class="multinomial",
        solver="lbfgs"
    )

    clf.fit(X_train, y_train)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    y_pred = clf.predict(X_val)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        y_pred,
        target_names=class_names,
        zero_division=0
    ))


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--task_dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet50"])
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    main(args)
