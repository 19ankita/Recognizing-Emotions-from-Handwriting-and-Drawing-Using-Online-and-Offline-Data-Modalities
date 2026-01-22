import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from src.utils.splits import train_test_split_ids
from src.utils.feature_config import ONLINE_FEATURES


# =========================
# CONFIG
# =========================
TASKS = ["cdt", "cursive_writing", "house", "pentagon", "words"]
TARGETS = ["stress", "anxiety", "depression"]

PSEUDO_TRAJ_ROOT = "pseudo_trajectory_experiments/data/pseudo_trajectories"
ONLINE_FEATURE_ROOT = "pseudo_trajectory_experiments/data/online_features"
OUT_ROOT = "pseudo_trajectory_experiments/results/pseudo"

os.makedirs(OUT_ROOT, exist_ok=True)


# =========================
# RUN ONE TASK FOR ONE LABEL
# =========================
def run_task(task, target):
    print(f"[PSEUDO] Task: {task} | Target: {target}")

    traj_dir = os.path.join(PSEUDO_TRAJ_ROOT, task)
    
    # Load pseudo features
    pseudo_df = pd.read_csv(
        os.path.join(PSEUDO_TRAJ_ROOT, f"{task}_pseudo_features.csv")
    )

    # Load labels from ONLINE CSV
    labels_df = pd.read_csv(
        os.path.join(ONLINE_FEATURE_ROOT, f"{task}_features.csv")
    )[["id", target]]

    # Merge with emotion labels
    df = pseudo_df.merge(labels_df, on="id", how="inner").dropna()

    # Train / test split (same logic as online)
    train_ids, test_ids = train_test_split_ids(df["id"].unique())

    train_df = df[df["id"].isin(train_ids)]
    test_df  = df[df["id"].isin(test_ids)]

    X_train = train_df[ONLINE_FEATURES].values
    y_train = train_df[target].values

    X_test  = test_df[ONLINE_FEATURES].values
    y_test  = test_df[target].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "task": task,
        "target": target,
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }


# =========================
# MAIN
# =========================
def main():
    results = []

    for target in TARGETS:
        for task in TASKS:
            res = run_task(task, target)
            results.append(res)

    out_path = os.path.join(OUT_ROOT, "pseudo_all_targets_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)

    print(f"\n Saved PSEUDO results for all targets → {out_path}")


if __name__ == "__main__":
    main()
