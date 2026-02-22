import os
import numpy as np
import pandas as pd


# ==========================================================
# TRAJECTORY-LEVEL FEATURE EXTRACTION
# ==========================================================

def extract_pseudo_features(traj: np.ndarray):
    """
    traj: np.ndarray shape (T, D), expects D>=3
    Columns: x, y, t

    Returns:
        dict of geometric + temporal + in-air features
    """

    # -------------------------
    # Safety checks
    # -------------------------
    if traj is None or traj.ndim != 2 or traj.shape[1] < 3 or traj.shape[0] < 4:
        return None

    x = traj[:, 0].astype(np.float64)
    y = traj[:, 1].astype(np.float64)
    t = traj[:, 2].astype(np.float64)

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)

    dt[dt <= 0] = 1e-6  # avoid division problems

    step_dist = np.sqrt(dx**2 + dy**2)

    # ==========================================================
    # GEOMETRIC FEATURES
    # ==========================================================

    path_length = float(step_dist.sum())

    displacement = float(
        np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    )
    straightness = float(displacement / (path_length + 1e-8))

    angles = np.arctan2(dy, dx)

    dominant_angle = float(
        np.arctan2(np.mean(np.sin(angles)),
                   np.mean(np.cos(angles)))
    )

    direction_concentration = float(
        np.sqrt(np.mean(np.cos(angles))**2 +
                np.mean(np.sin(angles))**2)
    )

    width = float(x.max() - x.min())
    height = float(y.max() - y.min())
    aspect_ratio = float(width / (height + 1e-8))

    median_step_length = float(np.median(step_dist))
    p95_step_length = float(np.percentile(step_dist, 95))

    # ==========================================================
    # TRUE TEMPORAL FEATURES
    # ==========================================================

    total_duration = float(t[-1] - t[0])

    speed = step_dist / dt
    mean_speed = float(np.mean(speed))
    std_speed = float(np.std(speed))

    accel = np.diff(speed) / dt[:-1]
    mean_acceleration = float(np.mean(accel)) if accel.size > 0 else 0.0
    std_acceleration = float(np.std(accel)) if accel.size > 0 else 0.0

    jerk = np.diff(accel) / dt[:-2] if accel.size > 1 else np.array([])
    mean_jerk = float(np.mean(np.abs(jerk))) if jerk.size > 0 else 0.0
    smoothness_index = float(np.sum(jerk**2)) if jerk.size > 0 else 0.0

    # ==========================================================
    # ESTIMATED IN-AIR FEATURES (Pause Proxy)
    # ==========================================================

    gap_threshold = np.percentile(dt, 90)
    pause_mask = dt > gap_threshold

    estimated_in_air_time = float(np.sum(dt[pause_mask]))
    number_of_pauses = int(np.sum(pause_mask))

    in_air_ratio = float(
        estimated_in_air_time / (total_duration + 1e-8)
    )

    mean_pause_duration = (
        float(estimated_in_air_time / number_of_pauses)
        if number_of_pauses > 0 else 0.0
    )

    # ==========================================================
    # RETURN DICTIONARY
    # ==========================================================

    return {
        # Original pseudo features
        "path_length": path_length,
        "straightness": straightness,
        "dominant_angle": dominant_angle,
        "direction_concentration": direction_concentration,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "median_step_length": median_step_length,
        "p95_step_length": p95_step_length,

        # Temporal features
        "total_duration": total_duration,
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "mean_acceleration": mean_acceleration,
        "std_acceleration": std_acceleration,
        "mean_jerk": mean_jerk,
        "smoothness_index": smoothness_index,

        # In-air related
        "estimated_in_air_time": estimated_in_air_time,
        "in_air_ratio": in_air_ratio,
        "number_of_pauses": number_of_pauses,
        "mean_pause_duration": mean_pause_duration,
    }


# ==========================================================
# FOLDER-LEVEL RUNNER (USED BY main.py)
# ==========================================================

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