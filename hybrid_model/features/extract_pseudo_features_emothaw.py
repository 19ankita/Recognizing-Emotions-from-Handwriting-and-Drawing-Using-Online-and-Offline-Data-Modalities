import numpy as np


def run_pseudo_feature_extraction(traj: np.ndarray):
    """
    traj: np.ndarray shape (T, D), expects D>=3 for temporal features
    Columns: x, y, t
    Returns dict with geometric + online + temporal features.
    """

    # --- safety checks ---
    if traj is None or traj.ndim != 2 or traj.shape[1] < 3 or traj.shape[0] < 4:
        return None

    x = traj[:, 0].astype(np.float64)
    y = traj[:, 1].astype(np.float64)
    t = traj[:, 2].astype(np.float64)

    # --------------------------------------------------
    # Basic differences
    # --------------------------------------------------
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)

    dt[dt <= 0] = 1e-6  # prevent division issues

    step_dist = np.sqrt(dx**2 + dy**2)
    path_length = float(step_dist.sum())

    # --------------------------------------------------
    # GEOMETRIC FEATURES 
    # --------------------------------------------------
    displacement = float(np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2))
    straightness = float(displacement / (path_length + 1e-8))

    angles = np.arctan2(dy, dx)
    dominant_angle = float(np.arctan2(np.mean(np.sin(angles)),
                                      np.mean(np.cos(angles))))

    direction_concentration = float(
        np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2)
    )

    width = float(x.max() - x.min())
    height = float(y.max() - y.min())
    aspect_ratio = float(width / (height + 1e-8))

    median_speed_proxy = float(np.median(step_dist))
    p95_speed_proxy = float(np.percentile(step_dist, 95))

    # --------------------------------------------------
    # TRUE TEMPORAL FEATURES
    # --------------------------------------------------
    total_duration = float(t[-1] - t[0])

    speed = step_dist / dt
    mean_speed = float(np.mean(speed))
    std_speed = float(np.std(speed))

    # Acceleration
    accel = np.diff(speed) / dt[:-1]
    mean_acceleration = float(np.mean(accel)) if accel.size > 0 else 0.0
    std_acceleration = float(np.std(accel)) if accel.size > 0 else 0.0

    # Jerk
    jerk = np.diff(accel) / dt[:-2] if accel.size > 1 else np.array([])
    mean_jerk = float(np.mean(np.abs(jerk))) if jerk.size > 0 else 0.0
    smoothness_index = float(np.sum(jerk**2)) if jerk.size > 0 else 0.0

    # --------------------------------------------------
    # ESTIMATED IN-AIR FEATURES (PAUSE-BASED PROXY)
    # --------------------------------------------------

    gap_threshold = np.percentile(dt, 90)
    pause_mask = dt > gap_threshold

    estimated_in_air_time = float(np.sum(dt[pause_mask]))
    in_air_ratio = float(estimated_in_air_time / (total_duration + 1e-8))
    number_of_pauses = int(np.sum(pause_mask))

    mean_pause_duration = (
        float(estimated_in_air_time / number_of_pauses)
        if number_of_pauses > 0 else 0.0
    )

    # --------------------------------------------------
    # FINAL FEATURE DICTIONARY
    # --------------------------------------------------

    return {
        # Original pseudo features
        "path_length": path_length,
        "straightness": straightness,
        "dominant_angle": dominant_angle,
        "direction_concentration": direction_concentration,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "median_step_length": median_speed_proxy,
        "p95_step_length": p95_speed_proxy,

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