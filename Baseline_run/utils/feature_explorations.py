from src.svc_reader import read_all_svc_files
from src.feature_utils import extract_features, segment_lines

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    
def plot_trajectory(df, title="Trajectory"):
    x = df["x"].values
    y = df["y"].values
    pen_down = df["pen_status"].values == 1

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, color="gray", alpha=0.3, label="Trajectory")
    plt.scatter(x[pen_down], y[pen_down], s=3, label="Pen-down")
    plt.scatter(x[~pen_down], y[~pen_down], s=3, label="Pen-up")

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.legend()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_baselines(df, lines, title="Baselines"):
    plt.figure(figsize=(6, 6))

    for x_l, y_l in lines:
        plt.plot(x_l, y_l, "k.", alpha=0.6)

        if len(x_l) > 5:
            a, b = np.polyfit(x_l, y_l, 1)
            xs = np.linspace(x_l.min(), x_l.max(), 100)
            ys = a * xs + b
            plt.plot(xs, ys, "r-", linewidth=2)

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(title)
    plt.show()    


def feature_explore():
    
    data = read_all_svc_files("Baseline_run/dataset/words")

    sample_id, arr = next(iter(data.items()))
    df = pd.DataFrame(arr, columns=["x", "y", "timestamp", "pen_status", "azimuth", "altitude", "pressure"])

    df["id"] = sample_id

    features = extract_features(df)

    for k,v in features.items():
        print(f"{k:30s} : {v:.4f}")
        
    plot_trajectory(df, title="Word – Trajectory")

    pen_down = df["pen_status"].values == 1
    lines = segment_lines(df["x"].values, df["y"].values, pen_down)
    plot_baselines(df, lines, title="Word – Baselines")
