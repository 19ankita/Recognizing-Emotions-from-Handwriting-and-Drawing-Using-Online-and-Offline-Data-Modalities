import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np



CLASS_NAMES = [
    "Normal",
    "Mild",
    "Moderate",
    "Severe",
    "Extremely Severe"
]


def plot_confusion_matrix(y_true, y_pred, output_path="outputs/confusion_matrix.png"):
    """
    Plot and save confusion matrix.
    """

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix – Depression Severity Classification")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace(".png", ".pdf"))
    plt.close()

    print(f"Saved confusion matrix → {output_path}")
    


def save_classification_report(y_true, y_pred,
                               output_csv="outputs/classification_report.csv"):
    """
    Save precision / recall / F1-score table.
    """

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    df = pd.DataFrame(report).transpose()
    df.to_csv(output_csv)

    print(f"Saved classification report → {output_csv}")

    return df

