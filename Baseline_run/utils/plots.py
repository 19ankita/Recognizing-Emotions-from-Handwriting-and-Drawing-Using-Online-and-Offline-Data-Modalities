import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("figures", exist_ok=True)

# -------------------------
# 0. Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------
# 1. Load and Clean Data
# -------------------------
csv_path = os.path.join(RESULTS_DIR, "Model_summary.csv")
df = pd.read_csv(csv_path, sep=";")

df.columns = df.columns.str.strip().str.lower()

for col in ["r2", "cv_mse", "cv_std"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["r2"])
df["cv_std"] = df["cv_std"].fillna(0)

# Keep only subscales (exclude TOTAL DASS)
df = df[df["target"].str.lower().isin(["depression", "anxiety", "stress"])]

# -------------------------
# 2. Define Experimental Mode
# -------------------------
df = df.sort_values(by=["task", "model", "target"]).reset_index(drop=True)

df["experimental_mode"] = (
    df.groupby(["task", "model", "target"]).cumcount()
    .map({0: "Joint / Aggregated", 1: "Subscale-Specific"})
)

# -------------------------
# 3. Aggregate (mean across tasks if needed)
# -------------------------
plot_df = (
    df.groupby(["target", "experimental_mode"], as_index=False)
    .agg(mean_r2=("r2", "mean"))
)

# -------------------------
# 4. Bar Plot
# -------------------------
sns.set(style="whitegrid", palette="Set2")

plt.figure(figsize=(9, 6))
sns.barplot(
    data=plot_df,
    x="target",
    y="mean_r2",
    hue="experimental_mode",
    errorbar=None
)

plt.axhline(0, color="black", linewidth=1, linestyle="--")
plt.xlabel("DASS Subscale")
plt.ylabel("R² Score")
plt.title("Comparison of Joint vs Subscale-Specific Modeling (R²)")
plt.legend(title="Experimental Mode")
plt.tight_layout()

plt.savefig("figures/barplot_r2_experimental_modes.pdf", dpi=300)
plt.close()
