import os
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import argparse


def plot_class_distribution(dataset_path):

    dataset = ImageFolder(dataset_path)
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    
    sns.set_theme(style="whitegrid", font="serif", font_scale=1.2)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    
    # Sorted class names
    class_names = [dataset.classes[i] for i in counts.keys()]
    values = [counts[i] for i in counts.keys()]
    total = sum(values)
    percentages = [v / total * 100 for v in values]

    print("\n=== CLASS DISTRIBUTION ===")
    for cls, count in counts.items():
        print(f"Class {cls}: {count} samples")


    # ===========================================================================
    #                 BAR CHART (COUNTS)
    # ===========================================================================
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=class_names, y=values, palette="Blues_d")

    plt.title("Class Distribution", fontsize=16, weight="bold")
    plt.xlabel("Class", fontsize=14)
    plt.ylabel("Sample Count", fontsize=14)

    # Add counts on bar tops
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=12)

    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/class_distribution.png")
    
    plt.show()

    # ===========================================================================
    #                 PIE CHART
    # ===========================================================================
    plt.figure(figsize=(8, 8))
    colors = sns.color_palette("Blues", len(values))

    plt.pie(
        values,
        labels=[f"{cls} ({p:.1f}%)" for cls, p in zip(class_names, percentages)],
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 12},
        colors=colors
    )

    plt.title("Class Distribution (Percentage)", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(f"outputs/class_distribution_pie.pdf")
    plt.show()
    
    # ===========================================================================
    #             PERCENTAGE BAR CHART
    # ===========================================================================
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=class_names, y=percentages, palette="Blues")

    plt.title("Class Distribution (Percentage)", fontsize=16, weight="bold")
    plt.xlabel("Class", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=14)

    # Add percentage labels
    for i, p in enumerate(percentages):
        ax.text(i, p + 0.3, f"{p:.1f}%", ha="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"outputs/class_distribution_percentage.pdf")
    plt.show()

    print("Saved the figs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    args = parser.parse_args()

    plot_class_distribution(args.dataset_path)
