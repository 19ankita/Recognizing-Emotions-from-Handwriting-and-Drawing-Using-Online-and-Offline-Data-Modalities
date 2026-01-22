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
    x, y = traj[:, 0], traj[:, 1]
    
    if len(x) < 2:
        return None

    # Differences
    dx = np.diff(x)
    dy = np.diff(y)
    step_dist = np.sqrt(dx**2 + dy**2)

    # Path length
    path_length = step_dist.sum()

    # Speed statistics (assuming uniform dt)
    mean_speed = step_dist.mean()
    std_speed  = step_dist.std()

    # Straightness
    displacement = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    straightness = displacement / (path_length + 1e-8)

    # Direction angles
    angles = np.arctan2(dy, dx)
    direction_variance = np.var(angles) if len(angles) > 0 else 0.0

    # Bounding box
    bbox_area = (x.max() - x.min()) * (y.max() - y.min())

    return {
        "path_length": path_length,
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "straightness": straightness,
        "direction_variance": direction_variance,
        "bbox_area": bbox_area,
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

        print(f"  Saved → {out_csv}")


if __name__ == "__main__":
    main()