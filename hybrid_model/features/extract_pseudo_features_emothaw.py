import os
import numpy as np
import pandas as pd


def extract_pseudo_features(traj: np.ndarray):
    """
    traj: np.ndarray shape (T, D), D>=2. Uses columns 0,1 as x,y.
    Returns dict with the 9 pseudo numeric features.
    """
    # --- safety checks ---
    if traj is None or traj.ndim != 2 or traj.shape[1] < 2 or traj.shape[0] < 2:
        return None

    x = traj[:, 0].astype(np.float64)
    y = traj[:, 1].astype(np.float64)

    # Step-wise differences
    dx = np.diff(x)
    dy = np.diff(y)
    step_dist = np.sqrt(dx**2 + dy**2)

    # If step_dist is empty (shouldn't happen because T>=2), guard anyway
    if step_dist.size == 0:
        return None

    # -------------------------
    # Path-based features
    # -------------------------
    path_length = float(step_dist.sum())

    displacement = float(np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2))
    straightness = float(displacement / (path_length + 1e-8))

    # -------------------------
    # Directional features
    # -------------------------
    angles = np.arctan2(dy, dx)  # radians

    dominant_angle = float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))

    R = float(np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2))
    direction_concentration = R  # [0, 1]

    # -------------------------
    # Spatial layout features
    # -------------------------
    width = float(x.max() - x.min())
    height = float(y.max() - y.min())
    aspect_ratio = float(width / (height + 1e-8))

    # -------------------------
    # Speed proxy statistics
    # -------------------------
    median_speed = float(np.median(step_dist))
    p95_speed = float(np.percentile(step_dist, 95))

    return {
        "path_length": path_length,
        "straightness": straightness,
        "dominant_angle": dominant_angle,
        "direction_concentration": direction_concentration,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "median_speed": median_speed,
        "p95_speed": p95_speed,
    }


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

            # IMPORTANT: keep 'id' for merging later
            feats["id"] = fname.replace(".npy", "")
            rows.append(feats)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out_csv, index=False)
            print(f"[INFO] Saved pseudo features to {out_csv} ({len(df)} samples)")
        else:
            print(f"[WARNING] No valid trajectories for {task}")