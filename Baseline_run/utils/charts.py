import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("figures", exist_ok=True)

# -------------------------
# 1. Load and Clean Data
# -------------------------
df = pd.read_csv("E:/2nd_thesis/Results/model_summary_latest.csv", sep=";")

# Standardize column names
df.columns = df.columns.str.strip().str.lower()
print("Cleaned column names:", df.columns.tolist())

# Convert numeric columns safely
for col in ["r2", "cv_mse", "cv_std"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows where R² is missing
df = df.dropna(subset=["r2"])

df["cv_std"] = df["cv_std"].fillna(0)

# List all unique targets
targets = df["target"].str.lower().unique().tolist()
print("Detected targets:", targets)

# Style
sns.set(style="whitegrid", palette="Set2")

# -------------------------
# 2. Loop Through Targets
# -------------------------
for target in targets:
    df_sub = df[df["target"].str.lower() == target]

    # =========================
    # BAR PLOT
    # =========================
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_sub, x="task", y="r2", hue="model", errorbar=None)
    plt.title(f"Model Performance (R²) for {target.capitalize()} Prediction Across Tasks")
    plt.ylabel("R² Score")
    plt.xlabel("Task")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join("figures", f"barplot_{target.replace(' ', '_')}.pdf")
    plt.savefig(save_path, dpi=100)
    plt.close()

    # =========================
    # HEATMAP
    # =========================
    heatmap_data = df_sub.pivot_table(
        index="task", columns="model", values="r2", aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", center=0, fmt=".2f")
    plt.title(f"Heatmap of R² Scores for {target.capitalize()} Prediction")
    plt.ylabel("Task")
    plt.xlabel("Model")
    plt.tight_layout()
    save_path = os.path.join("figures", f"heatmap_{target.replace(' ', '_')}.pdf")
    plt.savefig(save_path, dpi=100)
    plt.close()

    