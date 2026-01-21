import numpy as np


def extract_features(traj):
    x, y = traj[:, 0], traj[:, 1]

    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2)

    features = {
        "path_length": np.sum(speed),
        "mean_speed": np.mean(speed),
        "std_speed": np.std(speed),
        "max_speed": np.max(speed),
        "straightness": np.linalg.norm([x[-1]-x[0], y[-1]-y[0]]) / (np.sum(speed) + 1e-8)
    }

    return features
