import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from src.pseudo_features import extract_pseudo_dynamic_features


FEATURE_NAMES = [
    "Stroke Length",
    "Mean Thickness",
    "Stroke Segments",
    "Aspect Ratio",
    "Curvature"
]

# --------------------------------------------------
# Utility
# --------------------------------------------------
def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------
# Core function (IMPORTABLE from run_train.py)
# --------------------------------------------------
def visualize_single_image(image_path, save_dir=None):
    """
    Visualize pseudo-dynamic features for ONE image.
    Can be called from run_train.py or via CLI.
    """
    img = load_image(image_path)
    features = extract_pseudo_dynamic_features(img)

    save_path = None
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "single_image_pseudo_features.pdf"

    plot_single_image(img, features, save_path)


# --------------------------------------------------
# Plot helper
# --------------------------------------------------
def plot_single_image(img, features, save_path=None):
    plt.figure(figsize=(10, 4))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Handwriting Sample")
    plt.axis("off")

    # Feature vector
    plt.subplot(1, 2, 2)
    plt.bar(FEATURE_NAMES, features)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Normalized Value")
    plt.title("Pseudo-Dynamic Feature Vector")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()