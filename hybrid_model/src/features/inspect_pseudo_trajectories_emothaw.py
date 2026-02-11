import os
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
TASKS = ["cdt", "cursive_writing", "house", "pentagon", "words"]

TRAJ_ROOT = "data/EMOTHAW/pseudo_trajectories"
OUT_ROOT  = "data/EMOTHAW/pseudo_trajectories"

os.makedirs(OUT_ROOT, exist_ok=True)

# =========================
# FEATURE EXTRACTION
# =========================
def extract_pseudo_features(traj):
    """
    traj: np.ndarray of shape (T, D), D >= 2
    """
    x = traj[:, 0]
    y = traj[:, 1]

    if len(x) < 2:
        return None

    # ----------------------------------
    # Step-wise differences
    # ----------------------------------
    dx = np.diff(x)
    dy = np.diff(y)
    step_dist = np.sqrt(dx**2 + dy**2)

    # ----------------------------------
    # Path-based features
    # ----------------------------------
    path_length = step_dist.sum()

    displacement = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    straightness = displacement / (path_length + 1e-8)

    # ----------------------------------
    # Directional features
    # ----------------------------------
    angles = np.arctan2(dy, dx)

    # Circular mean (dominant direction)
    dominant_angle = np.arctan2(
        np.mean(np.sin(angles)),
        np.mean(np.cos(angles))
    )

    # Circular variance → concentration
    R = np.sqrt(
        np.mean(np.cos(angles))**2 +
        np.mean(np.sin(angles))**2
    )
    direction_concentration = R  # in [0, 1]

    # ----------------------------------
    # Spatial layout features
    # ----------------------------------
    width = x.max() - x.min()
    height = y.max() - y.min()
    aspect_ratio = width / (height + 1e-8)

    # ----------------------------------
    # Speed statistics (proxy)
    # ----------------------------------
    median_speed = np.median(step_dist)
    p95_speed = np.percentile(step_dist, 95)

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



# =========================
# MAIN LOOP
# =========================
def main():
    
    for task in TASKS:
        print(f"\n[INFO] Extracting pseudo-features for: {task}")

        traj_dir = os.path.join(TRAJ_ROOT, task)
        out_csv  = os.path.join(OUT_ROOT, f"{task}_pseudo_features.csv")
        
        if not os.path.isdir(traj_dir):
            print(f"[WARNING] Missing directory: {traj_dir}")
            continue

        rows = []

        for fname in sorted(os.listdir(traj_dir)):
            if not fname.endswith(".npy"):
                continue

            traj = np.load(os.path.join(traj_dir, fname))

            if traj.ndim != 2 or traj.shape[1] < 2:
                continue

            feats = extract_pseudo_features(traj)
            if feats is None:
                continue
            
            feats["sample_id"] = fname.replace(".npy", "")
            rows.append(feats)

        if not rows:
            print(f"[WARNING] No valid trajectories for {task}")
            continue
        
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)

        print(f"  Saved to {out_csv}")


if __name__ == "__main__":
    main()