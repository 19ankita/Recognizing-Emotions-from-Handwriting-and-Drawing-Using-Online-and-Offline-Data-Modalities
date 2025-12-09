import os
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from collections import Counter


def plot_class_distribution(dataset_path):

    dataset = ImageFolder(dataset_path)
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)

    print("\n=== CLASS DISTRIBUTION ===")
    for cls, count in counts.items():
        print(f"Class {cls}: {count} samples")

    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Class Index")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.show()
