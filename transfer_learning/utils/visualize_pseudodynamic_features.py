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


def parse_args():
    parser = argparse.ArgumentParser("Pseudo-Dynamic Feature Visualization")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to handwriting image")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional path to save figure as PDF/PNG")
    return parser.parse_args()


def main(args):
    # Load image
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract features
    features = extract_pseudo_dynamic_features(img)

    # Plot
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

    if args.save:
        plt.savefig(args.save, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
