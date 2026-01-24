import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("figures", exist_ok=True)

BASELINE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
RESULTS_DIR = os.path.join(BASELINE_DIR, "results")
csv_path = os.path.join(RESULTS_DIR, "model_summary.csv")

df = pd.read_csv(csv_path, sep=";")
df.columns = df.columns.str.strip().str.lower()

for col in ["r2", "cv_mse", "cv_std"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["r2"])

df = df[
    (df["target"].isin(["depression", "anxiety", "stress"])) &
    (df["mode"].isin(["multi-output", "subscales"]))
]

df["experimental_mode"] = df["mode"].map({
    "multi-output": "Joint / Aggregated",
    "subscales": "Subscale-Specific"
})

# Aggregate across MODELS only
plot_df = (
    df.groupby(["task", "target", "experimental_mode"], as_index=False)
      .agg(mean_r2=("r2", "mean"))
)

sns.set(style="whitegrid", palette="Set2")

g = sns.catplot(
    data=plot_df,
    x="task",
    y="mean_r2",
    hue="experimental_mode",
    col="target",
    kind="bar",
    height=5,
    aspect=1.1,
    errorbar=None
)

g.set_axis_labels("Task", "Mean R²")
g.set_titles("{col_name}")
for ax in g.axes.flatten():
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig("figures/r2_joint_vs_subscale_per_task.pdf", dpi=300)
plt.close()
