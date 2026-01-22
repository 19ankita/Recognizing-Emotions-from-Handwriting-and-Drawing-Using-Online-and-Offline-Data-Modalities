import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


# ===================== PATH SETUP  =====================
base_dir = os.path.dirname(os.path.abspath(__file__))

dataset_dir  = os.path.join(base_dir, "dataset")

# ===========================================================

TASK_DIRS = {
    "cdt": "dataset/cdt",
    "house": "dataset/house",
    "pentagon": "dataset/pentagon",
    "words": "dataset/words",
    "cursive_writing": "dataset/cursive_writing",
}


# =====================================================
# IO UTILITIES
# =====================================================
def read_svc_file(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f.readlines()[1:]:  # skip header
            parts = line.strip().split()
            if len(parts) != 7:
                continue
            data.append(tuple(map(int, parts)))
    return np.array(data)


def read_all_svc_files(folder):
    svc_files = glob.glob(os.path.join(folder, "*.svc")) + glob.glob(os.path.join(folder, "*.SVC"))
    svc_files.sort()

    dataset = {}
    for f in svc_files:
        sample_id = os.path.splitext(os.path.basename(f))[0]
        dataset[sample_id] = read_svc_file(f)

    return dataset


def extract_user_number(sample_id):
    match = re.search(r"u(\d+)", sample_id)
    return int(match.group(1)) if match else None


# =====================================================
# FEATURE UTILITIES
# =====================================================
def euclidean_dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def path_length(x, y):
    return np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))


def instantaneous_speed(x, y, t):
    dx, dy, dt = np.diff(x), np.diff(y), np.diff(t)
    valid = dt > 0
    return np.sqrt((dx[valid] / dt[valid]) ** 2 + (dy[valid] / dt[valid]) ** 2)


def acceleration(speeds, t):
    if len(speeds) < 2:
        return np.array([])
    dt = np.diff(t[1:])
    dv = np.diff(speeds)
    valid = dt > 0
    return dv[valid] / dt[valid]


def straightness(x, y):
    if len(x) < 2:
        return 0
    return euclidean_dist(x[0], y[0], x[-1], y[-1]) / path_length(x, y)


def stop_ratio(speeds):
    if len(speeds) == 0:
        return 0
    eps = 0.1 * np.median(speeds[speeds > 0]) if np.any(speeds > 0) else 0
    return np.sum(speeds < eps) / len(speeds) if eps > 0 else 0


# =====================================================
# MAIN FEATURE EXTRACTOR
# =====================================================
def extract_features(df):
    x, y = df["x"].values, df["y"].values
    t = df["timestamp"].values
    p = df["pressure"].values
    status = df["pen_status"].values

    pen_down = status == 1
    pen_up = ~pen_down
    dt = np.diff(t, prepend=t[0])

    features = {}

    # -------- Temporal --------
    features["F1_in_air_time"] = np.sum(dt[pen_up])
    features["F2_on_paper_time"] = np.sum(dt[pen_down])
    features["F3_total_time"] = t[-1] - t[0]
    features["F4_stroke_count"] = np.sum(np.diff(status) == 1) + 1
    features["duty_cycle"] = (
        features["F2_on_paper_time"] / features["F3_total_time"]
        if features["F3_total_time"] > 0 else 0
    )

    # -------- Kinematic --------
    speeds = instantaneous_speed(x, y, t)
    speeds_down = speeds[pen_down[1:]] if len(speeds) else []

    features["path_length"] = path_length(x, y)
    acc = acceleration(speeds, t)
    features["median_acceleration"] = np.median(acc) if len(acc) else 0
    features["median_speed"] = np.median(speeds_down) if len(speeds_down) else 0
    features["p95_speed"] = np.percentile(speeds_down, 95) if len(speeds_down) else 0
    features["stop_ratio"] = stop_ratio(speeds_down)

    # -------- Geometric --------
    features["straightness"] = straightness(x, y)
    dx, dy = np.diff(x), np.diff(y)

    if len(dx):
        thetas = np.arctan2(dy, dx)
        C, S = np.cos(thetas).sum(), np.sin(thetas).sum()
        features["dominant_angle"] = np.degrees(np.arctan2(S, C))
        features["direction_concentration"] = (C**2 + S**2) / len(thetas)
    else:
        features["dominant_angle"] = 0
        features["direction_concentration"] = 0

    # -------- Pressure --------
    features["mean_pressure"] = np.mean(p[pen_down]) if np.any(pen_down) else 0

    # -------- Bounding box --------
    if np.any(pen_down):
        W = x[pen_down].max() - x[pen_down].min()
        H = y[pen_down].max() - y[pen_down].min()
    else:
        W, H = 0, 0

    features["width"] = W
    features["height"] = H
    features["aspect_ratio"] = W / H if H > 0 else 0

    # -------- Ink density --------
    try:
        hull = ConvexHull(np.vstack((x[pen_down], y[pen_down])).T)
        area = hull.volume
    except:
        area = 0

    features["ink_density"] = features["path_length"] / area if area > 0 else 0

    return features


# =====================================================
# RUN ALL TASKS
# =====================================================

def run_all_tasks(task_dirs, out_dir):
    """
    task_dirs: dict {task_name: path_to_svc_folder}
    out_dir: where CSVs will be saved
    """
    os.makedirs(out_dir, exist_ok=True)

    for task_name, dataset_dir in task_dirs.items():
        data = read_all_svc_files(dataset_dir)
        rows = []

        for sample_id, arr in data.items():
            df = pd.DataFrame(
                arr,
                columns=[
                    "x", "y", "timestamp",
                    "pen_status", "azimuth",
                    "altitude", "pressure"
                ],
            )

            feats = extract_features(df)
            feats["id"] = sample_id
            feats["user"] = extract_user_number(sample_id)
            rows.append(feats)

        out_df = pd.DataFrame(rows)
        out_df.to_csv(
            os.path.join(out_dir, f"{task_name}_features.csv"),
            index=False
        )


def main():
    
    for task_name, dataset_dir in TASK_DIRS.items():
        data = read_all_svc_files(dataset_dir)

        if len(data) == 0:
            raise RuntimeError(
                f"No .svc files found for task '{task_name}' in {dataset_dir}"
        )


if __name__ == "__main__":
    main()
