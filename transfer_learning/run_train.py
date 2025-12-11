import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os
import math

from src.dataset import get_dataloaders
from src.model import build_resnet18, build_resnet50
from src.utils import accuracy, save_checkpoint

# GradCAM library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2

torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------
# Warmup + Cosine LR Scheduler
# ------------------------------------------------------------
def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        # Warmup phase
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        # Cosine decay
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------
# GradCAM visualization function
# ------------------------------------------------------------
def generate_gradcam(model, image_tensor, pseudo_tensor, save_path, layer_name):
    model.eval()

    # pick target layer
    target_layer = dict(model.backbone.named_children())[layer_name]

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=True)

    grayscale_cam = cam(
        input_tensor=[image_tensor],
        aug_smooth=True,
        eigen_smooth=True
    )[0]  # take batch index 0

    # Convert tensor to numpy image
    img = image_tensor.cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())

    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    
    
    
# ------------------------------------------------------------
# Training Function
# ------------------------------------------------------------
def run_train(args):

    # Prepare output directory
    os.makedirs("outputs", exist_ok=True)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device → {device}")

    # --------------------------------------------------------
    # Load dataset(s)
    # --------------------------------------------------------
    train_loader, val_loader, num_classes = get_dataloaders(
        task=args.task,
        task_root=args.task_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio
    )
    
    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------
    print(f"\n>>> Building model: {args.model}")
    if args.model == "resnet18":
        model = build_resnet18(num_classes, freeze_backbone=args.freeze_backbone)
    elif args.model == "resnet50":
        model = build_resnet50(num_classes, freeze_backbone=args.freeze_backbone)
    else:
        raise ValueError("Unknown model: choose resnet18 or resnet50")

    model = model.to(device)


    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (Phase 1: classifier-only or full model depending on freeze_backbone)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    # Scheduler for Phase 1
    scheduler = get_scheduler(
        optimizer,
        warmup_epochs=2,
        total_epochs=args.epochs
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler(device="cuda")

    # Training history storage
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    best_acc = 0

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(args.epochs):
        print(f"\n==== Epoch {epoch+1}/{args.epochs} ====")

        # --------------------------------------------------------
        # PHASE SWITCH: Unfreeze Backbone at Epoch 10
        # --------------------------------------------------------
        if epoch == 3:
            print("\n>>> Unfreezing backbone for fine-tuning...")

            # Unfreeze ALL parameters
            model.requires_grad_(True)

            # Rebuild optimizer for full fine-tuning
            optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-5,              # very low LR for fine-tuning
                weight_decay=1e-4
            )

            # Rebuild scheduler for the remaining epochs
            scheduler = get_scheduler(
                optimizer,
                warmup_epochs=0,
                total_epochs=args.epochs - epoch
            )

        # --------------------------------------------------------
        # TRAINING PHASE
        # --------------------------------------------------------
        model.train()
        train_loss_total = 0
        train_acc_total = 0

        batch = next(iter(train_loader))
        print("Batch len:", len(batch))
        print("Types:", type(batch))
        print("Contents:", batch)
        exit()

        for images, pseudo, labels in tqdm(train_loader, desc="Training"):
            images = images.to(device)  
            pseudo = pseudo.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                pseudo = pseudo.to(device)  # float tensor
                outputs = model(images, pseudo)
                loss = criterion(outputs, labels)

            # Backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()
            train_acc_total += accuracy(outputs, labels)

        train_loss = train_loss_total / len(train_loader)
        train_acc = train_acc_total / len(train_loader)

        # --------------------------------------------------------
        # VALIDATION PHASE
        # --------------------------------------------------------
        model.eval()
        val_loss_total = 0
        val_acc_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(device)  
                pseudo = pseudo.to(device)
                labels = labels.to(device)

                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss_total += loss.item()
                val_acc_total += accuracy(outputs, labels)

        val_loss = val_loss_total / len(val_loader)
        val_acc = val_acc_total / len(val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # --------------------------------------------------------
        # Logging + Scheduler Update
        # --------------------------------------------------------
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print("Current LR:", current_lr)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # --------------------------------------------------------
        # Save best model
        # --------------------------------------------------------
        if val_acc > best_acc:
            best_acc = val_acc
            
            filename = f"best_model_{args.task}_{args.model}.pth"
            save_path = os.path.join("outputs", filename)

            save_checkpoint(model, save_path)
            print(f"Saved new BEST model : {filename}")
                
            print("Saved new BEST model")
            
            
        # ------------------------------------------------------------
        # GradCAM Visualization for best model
        # ------------------------------------------------------------
        sample_image, sample_pseudo, _ = next(iter(val_loader))
        sample_image = sample_image[0].to(device)
        sample_pseudo = sample_pseudo[0].to(device)

        save_path = f"outputs/gradcam/epoch_{epoch+1}_best_cam.jpg"
        generate_gradcam(
            model=model,
            image_tensor=sample_image.unsqueeze(0),
            pseudo_tensor=sample_pseudo.unsqueeze(0),
            save_path=save_path,
            layer_name="layer4"  # most interpretable layer
        )
        print(f"GradCAM saved → {save_path}")

    # Save history file
    with open("outputs/history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining complete! History saved to outputs/history.json")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True,
                        help="Select task: 1,2,3,4,5 or 'all' for all tasks combined")

    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"])
    
    parser.add_argument("--task_dir", type=str, required=True,
                        help="Path to your dataset root folder containing class subdirectories.")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for DataLoader.")

    args = parser.parse_args()

    run_train(args)