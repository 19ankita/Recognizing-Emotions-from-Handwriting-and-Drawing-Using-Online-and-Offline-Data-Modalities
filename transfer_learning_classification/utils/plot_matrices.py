import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

import pandas as pd

CLASS_NAMES = [
    "Normal",
    "Mild",
    "Moderate",
    "Severe",
    "Extremely\nSevere"
]

LABELS = list(range(len(CLASS_NAMES)))  # [0,1,2,3,4]

def plot_confusion_matrix(y_true, y_pred, state, output_dir="outputs"):
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
    plt.title(f"Confusion Matrix – {state.capitalize()} Severity Classification")

    plt.tight_layout()
    
    png_path = f"{output_dir}/confusion_matrix_{state}.png"
    pdf_path = png_path.replace(".png", ".pdf")

    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Saved confusion matrix to {png_path}")
    


def save_classification_report(y_true, y_pred, state, output_dir="outputs"):

    """
    Save precision / recall / F1-score table for a given emotional state.
    Missing classes are reported with zero scores.
    """

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    df = pd.DataFrame(report).transpose()
    
    csv_path = f"{output_dir}/classification_report_{state}.csv"
    df.to_csv(csv_path)

    print(f"Saved classification report to {csv_path}")

    return df

