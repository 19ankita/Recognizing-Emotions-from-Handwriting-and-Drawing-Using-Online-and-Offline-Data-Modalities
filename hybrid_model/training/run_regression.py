# training/run_regression.py
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

PSEUDO_FEATURES = [
    # geometric
    "path_length",
    "straightness",
    "width",
    "height",
    "aspect_ratio",

    # temporal
    "total_duration",
    "mean_speed",
    "std_speed",

    # in-air
    "estimated_in_air_time",
    "in_air_ratio",
]

def run_regression(merged_csv, task="cursive_writing", out_root="results/pseudo",
                   targets=("stress","anxiety","depression","total")):
    os.makedirs(out_root, exist_ok=True)
    df = pd.read_csv(merged_csv)

    results = []
    for target in targets:
        d = df.dropna(subset=PSEUDO_FEATURES + [target, "user"])
        X = d[PSEUDO_FEATURES].values
        y = d[target].values
        groups = d["user"].values

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        results.append({
            "task": task,
            "target": target,
            "r2": r2_score(y[test_idx], y_pred),
            "mse": mean_squared_error(y[test_idx], y_pred),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })

    out_path = os.path.join(out_root, f"{task}_pseudo_lr_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"[INFO] Saved regression results to {out_path}")
    return results