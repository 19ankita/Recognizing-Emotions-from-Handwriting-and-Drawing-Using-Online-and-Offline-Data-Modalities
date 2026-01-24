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

PSEUDO_TRAJ_ROOT = "data/EMOTHAW/pseudo_trajectories"
ONLINE_FEATURE_ROOT = "data/EMOTHAW/online_features"
LABEL_PATH = "labels/DASS_scores_clean.csv"

OUT_ROOT = "results/pseudo"
os.makedirs(OUT_ROOT, exist_ok=True)

PSEUDO_NUMERIC_FEATURES = [
    "path_length",
    "straightness",
    "dominant_angle",
    "direction_concentration",
    "width",
    "height",
    "aspect_ratio",
    "median_speed",
    "p95_speed",
]


# =========================
# RUN ONE TASK FOR ONE LABEL
# =========================
def run_task(task, target):
    print(f"[PSEUDO] Task: {task} | Target: {target}")
    
    # -------------------------
    # Load pseudo features
    # -------------------------
    pseudo_path = os.path.join(PSEUDO_TRAJ_ROOT, f"{task}_pseudo_features.csv")
    pseudo_df = pd.read_csv(pseudo_path, sep=";")
    
    online_path = os.path.join(ONLINE_FEATURE_ROOT, f"{task}_with_dass.csv")
    online_df = pd.read_csv(online_path , sep=";")
    
    # Normalize ID strings 
    pseudo_df["id"] = pseudo_df["id"].astype(str).str.strip()
    online_df["id"] = online_df["id"].astype(str).str.strip()

    df = (
        pseudo_df.merge(
            online_df[["id", "user", "depression", "anxiety", "stress", "total"]],
            on="id",
            how="inner"
        )
        .dropna()
    )
    
    # Save merged pseudo features with labels (per task)
    merged_out = os.path.join(
        OUT_ROOT, f"{task}_pseudo_with_dass.csv"
    )
    df.to_csv(merged_out, index=False, sep=";")
    print(f"[PSEUDO] Saved merged file to {merged_out}")
    
    # -------------------------
    # Fix numeric formatting in features
    # -------------------------

    for col in PSEUDO_NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)   # remove thousand separators
                .str.replace(",", ".", regex=False)  # handle decimal commas
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with invalid numeric values
    df = df.dropna(subset=PSEUDO_NUMERIC_FEATURES)

    # Train / test split (same logic as online)
    train_ids, test_ids = train_test_split_ids(df["id"].unique())

    train_df = df[df["id"].isin(train_ids)]
    test_df  = df[df["id"].isin(test_ids)]

    X_train = train_df[PSEUDO_NUMERIC_FEATURES].values
    y_train = train_df[target].values

    X_test  = test_df[PSEUDO_NUMERIC_FEATURES].values
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

    print(f"\n Saved PSEUDO results for all targets to {out_path}")


if __name__ == "__main__":
    main()
