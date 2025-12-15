
import os
import numpy as np
import torch

def get_class_names_from_task(task_root, task_name):
    """
    Get class names from a specific EMOTHAW task folder
    """
    task_path = os.path.join(task_root, task_name)

    classes = [
        d for d in os.listdir(task_path)
        if os.path.isdir(os.path.join(task_path, d))
    ]
    classes.sort()
    return classes

# ----------------------------------------------------
# HELPER: Get predictions
# ----------------------------------------------------
def get_predictions(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, pseudo, labels in loader:
            images, pseudo = images.to(device), pseudo.to(device)
            labels = labels.to(device)

            outputs = model(images, pseudo)
            preds = outputs.argmax(dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)