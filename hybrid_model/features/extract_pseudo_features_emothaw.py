# features/extract_pseudo_features_emothaw.py
import os
import numpy as np
import pandas as pd


def extract_pseudo_features(traj: np.ndarray):
    # --- safety checks ---
    if traj is None or traj.ndim != 2 or traj.shape[1] < 2 or traj.shape[0] < 2:
        return None

    x = traj[:, 0].astype(np.float64)
    y = traj[:, 1].astype(np.float64)

    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2)

    # speed length is T-1, so still can’t be empty because T>=2
    path_length = float(np.sum(speed))

    features = {
        "path_length": path_length,
        "mean_speed": float(np.mean(speed)),
        "std_speed": float(np.std(speed)),
        "max_speed": float(np.max(speed)),
        "straightness": float(
            np.linalg.norm([x[-1] - x[0], y[-1] - y[0]]) / (path_length + 1e-8)
        ),
    }
    return features


def run_pseudo_feature_extraction(
    tasks=("cursive_writing",),
    traj_root="data/processed/EMOTHAW/pseudo_trajectories",
    out_root="data/processed/EMOTHAW/pseudo_features",
):
    os.makedirs(out_root, exist_ok=True)

    for task in tasks:
        print(f"\n[INFO] Extracting pseudo-features for: {task}")

        traj_dir = os.path.join(traj_root, task)
        out_csv = os.path.join(out_root, f"{task}_pseudo_features.csv")

        if not os.path.isdir(traj_dir):
            print(f"[WARNING] Missing directory: {traj_dir}")
            continue

        rows = []
        for fname in sorted(os.listdir(traj_dir)):
            if not fname.endswith(".npy"):
                continue

            path = os.path.join(traj_dir, fname)
            try:
                traj = np.load(path)
            except Exception as e:
                print(f"[WARNING] Failed to load {path}: {e}")
                continue

            feats = extract_pseudo_features(traj)
            if feats is None:
                continue

            feats["id"] = fname.replace(".npy", "")
            rows.append(feats)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out_csv, index=False)
            print(f"[INFO] Saved pseudo features to {out_csv} ({len(df)} samples)")
        else:
            print(f"[WARNING] No valid trajectories for {task}")