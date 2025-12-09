import os
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from collections import Counter
import argparse


def plot_class_distribution(dataset_path):

    dataset = ImageFolder(dataset_path)
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)

    print("\n=== CLASS DISTRIBUTION ===")
    for cls, count in counts.items():
        print(f"Class {cls}: {count} samples")

    plt.figure(figsize=(10, 5))
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Class Index")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    args = parser.parse_args()

    plot_class_distribution(args.dataset_path)
