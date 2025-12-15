import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os

from sklearn.metrics import confusion_matrix, classification_report

from src.dataset import get_dataloaders
from src.model import build_resnet18, build_resnet50
from utils.helper import get_class_names_from_task


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------
def run_diagnostics(model, dataloader, device, class_names):

    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for images, pseudo, labels in dataloader:
            images = images.to(device)
            pseudo = pseudo.to(device)

            outputs = model(images, pseudo)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(outputs.cpu())

    y_true = torch.cat(all_labels)
    y_pred = torch.cat(all_preds)
    logits = torch.cat(all_logits)

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    ))

    # --------------------------------------------------
    # Mean logits per TRUE class
    # --------------------------------------------------
    print("\nMean logits per TRUE class:")
    for idx, name in enumerate(class_names):
        class_logits = logits[y_true == idx]
        if len(class_logits) > 0:
            print(f"{name}: {class_logits.mean(dim=0).numpy()}")

    # --------------------------------------------------
    # Mean softmax probabilities
    # --------------------------------------------------
    probs = F.softmax(logits, dim=1)
    print("\nMean softmax probabilities:")
    for idx, name in enumerate(class_names):
        print(f"{name}: {probs[y_true == idx].mean(dim=0).numpy()}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load validation data only
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

    # Run diagnostics
    run_diagnostics(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=class_names
    )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--task_dir", type=str, required=True)

    parser.add_argument("--model", type=str,
                        choices=["resnet18", "resnet50"],
                        required=True)

    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    main(args)
