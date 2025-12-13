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
    parser = argparse.ArgumentParser("Pseudo-Dynamic Feature Analysis")

    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single handwriting image")

    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset root with class subfolders")

    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save figures (PDF/PNG)")

    return parser.parse_args()


# --------------------------------------------------
# Utility
# --------------------------------------------------
def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------
# 1. Single-image visualization
# --------------------------------------------------
def plot_single_image(img, features, save_path=None):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Handwriting Sample")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.bar(FEATURE_NAMES, features)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Normalized Value")
    plt.title("Pseudo-Dynamic Feature Vector")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


# --------------------------------------------------
# 2. Dataset-level extraction
# --------------------------------------------------
def extract_dataset_features(data_dir):
    features_list = []
    labels = []
    class_names = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])

    for label, cls in enumerate(class_names):
        for img_path in Path(data_dir, cls).glob("*.png"):
            img = load_image(img_path)
            feat = extract_pseudo_dynamic_features(img)
            features_list.append(feat)
            labels.append(label)

    return np.array(features_list), np.array(labels), class_names


# --------------------------------------------------
# 3. Class-wise mean comparison
# --------------------------------------------------
def plot_class_means(features, labels, class_names, save_path=None):
    means = [features[labels == i].mean(axis=0) for i in range(len(class_names))]
    x = np.arange(len(FEATURE_NAMES))

    plt.figure(figsize=(8, 4))
    for i, mean in enumerate(means):
        plt.bar(x + i * 0.3, mean, width=0.3, label=class_names[i])

    plt.xticks(x + 0.3 * (len(class_names) - 1) / 2,
               FEATURE_NAMES, rotation=45, ha="right")
    plt.ylabel("Normalized Value")
    plt.title("Class-wise Pseudo-Dynamic Feature Comparison")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


# --------------------------------------------------
# 4. Feature boxplots
# --------------------------------------------------
def plot_boxplots(features, labels, class_names, save_path=None):
    plt.figure(figsize=(12, 4))

    for i, name in enumerate(FEATURE_NAMES):
        plt.subplot(1, len(FEATURE_NAMES), i + 1)
        data = [features[labels == j, i] for j in range(len(class_names))]
        plt.boxplot(data, labels=class_names)
        plt.title(name)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(args):

    if args.image:
        img = load_image(args.image)
        features = extract_pseudo_dynamic_features(img)

        save_path = None
        if args.save_dir:
            save_path = Path(args.save_dir) / "single_image_pseudo_features.pdf"

        plot_single_image(img, features, save_path)

    if args.data_dir:
        features, labels, class_names = extract_dataset_features(args.data_dir)

        save_mean = None
        save_box = None
        if args.save_dir:
            save_mean = Path(args.save_dir) / "classwise_feature_means.pdf"
            save_box = Path(args.save_dir) / "feature_boxplots.pdf"

        plot_class_means(features, labels, class_names, save_mean)
        plot_boxplots(features, labels, class_names, save_box)


if __name__ == "__main__":
    args = parse_args()
    main(args)
