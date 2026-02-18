import numpy as np
from scipy.spatial import ConvexHull

# ---------- Utility functions ----------
def euclidean_dist(x1, y1, x2,  y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def path_length(x, y):
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

def instantaneous_speed(x, y, t):
    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)
    
    valid = dt > 0
    vx = np.divide(dx[valid], dt[valid])
    vy = np.divide(dy[valid], dt[valid])
    
    return np.sqrt(vx**2 + vy**2)

def acceleration(speeds):
    if len(speeds) < 2:
        return np.array([])
    return np.diff(speeds)


def straightness(x, y):
    chord = euclidean_dist(x[0], y[0], x[-1], y[-1])
    return chord / path_length(x, y) if path_length(x, y) > 0 else 0

def stop_ratio(speeds):
    if len(speeds) == 0:
        return 0
    
    eps = 0.1 * np.median(speeds[speeds > 0]) if np.any(speeds > 0) else 0
    return np.sum(speeds < eps) / len(speeds) if eps > 0 else 0


# ---------- Main feature extractor ----------    
def extract_features(df):

    """
    Extract temporal, kinematic, geometric, pressure, slant, and pen-up features
    from a single handwriting trajectory.

    Input
    -----
    df : pandas.DataFrame
        A DataFrame containing one sample/trajectory with at least these columns:
        - 'x' (float): x-coordinate sequence
        - 'y' (float): y-coordinate sequence
        - 'timestamp' (float/int): time sequence (must be non-decreasing)
        - 'pen_status' (int): 1 = pen down, 0 = pen up
        - 'pressure' (float): pressure values (used only for pen-down points)
        Other columns may be present (e.g., azimuth, altitude, id) and are ignored.

    Output
    ------
    features : dict[str, float]
        Dictionary mapping feature names to scalar values:
        Temporal:
          - F1_in_air_time, F2_on_paper_time, F3_total_time, F4_stroke_count, duty_cycle
        Kinematic:
          - path_length, median_acceleration, median_speed, p95_speed, stop_ratio
        Geometric:
          - straightness, dominant_angle, direction_concentration
        Pressure:
          - mean_pressure
        Bounding box:
          - width, height, aspect_ratio
        Ink density:
          - ink_density (path_length / convex hull area of pen-down points)
        Slant:
          - mean_slant, std_slant, abs_mean_slant
        Pen-up:
          - pen_up_path_length, median_pen_up_speed, p95_pen_up_speed, pen_up_ratio

    Notes
    -----
    - Speed is computed from consecutive points where dt > 0.
    - "Acceleration" is computed as diff(speed) per step (not time-normalized).
    - Convex hull area is computed on pen-down points; if it fails or is zero,
      ink_density is set to 0.
    """  
    x = df["x"].values
    y = df["y"].values
    t = df["timestamp"].values
    p = df["pressure"].values
    status = df["pen_status"].values # 1 = pen down, 0 = pen up
    
    features = {}
    
    
    # --- Temporal ---
    pen_down = status == 1
    pen_up = ~pen_down
    dt = np.diff(t, prepend=t[0])
    
    features["F1_in_air_time"] = np.sum(dt[pen_up])
    features["F2_on_paper_time"] = np.sum(dt[pen_down])
    features["F3_total_time"] = t[-1] - t[0]
    features["F4_stroke_count"] = np.sum(np.diff(status) == 1) + 1
    
    
    # Duty cycle
    features["duty_cycle"] = (
        features["F2_on_paper_time"] / features["F3_total_time"]
        if features["F3_total_time"] > 0 else 0
    )
    
    
    # --- Kinematic ---
    speeds = instantaneous_speed(x, y, t)
    
    # define speeds_down
    if len(speeds):
        pen_down_pairs = pen_down[:-1] & pen_down[1:]
        speeds_down = speeds[pen_down_pairs[:len(speeds)]]
    else:
        speeds_down = np.array([])

    
    features["path_length"] = path_length(x, y)
    acc = acceleration(speeds)
    features["median_acceleration"] = np.median(acc) if len(acc) else 0

    features["median_speed"] = np.median(speeds_down) if len(speeds_down) else 0
    features["p95_speed"] = np.percentile(speeds_down, 95) if len(speeds_down) else 0
    features["stop_ratio"] = stop_ratio(speeds_down)  
    
    
    # --- Geometric ---      
    features["straightness"] = straightness(x, y)
    dx, dy = np.diff(x), np.diff(y)
    
    if len(dx) > 0:
        thetas = np.arctan2(dy, dx)
        C, S = np.cos(thetas).sum(), np.sin(thetas).sum()
        features["dominant_angle"] = np.degrees(np.arctan2(S, C))
        features["direction_concentration"] = (C**2 + S**2) / len(thetas)
    else:
        features["dominant_angle"] = 0
        features["direction_concentration"] = 0        
        
        
     # --- Pressure ---        
    features["mean_pressure"] = np.mean(p[pen_down]) if np.any(pen_down) else 0
     
    
    # --- Bounding box / aspect ratio ---
    if np.any(pen_down):
        W = x[pen_down].max() - x[pen_down].min()
        H = y[pen_down].max() - y[pen_down].min()
    else:
        W, H = 0, 0
    
    features["width"], features["height"] = W, H
    features["aspect_ratio"] = W / H if H > 0 else 0             
    
    
    # --- Ink density ---
    area = 0  
    try:
        hull = ConvexHull(np.vstack((x[pen_down], y[pen_down])).T)
        area = hull.volume
    except: 
        area = 0
    features["ink_density"] = features["path_length"] / area if area > 0 else 0  
    
    # --- Slant features ---
    if len(dx) > 0:
        slants = np.degrees(np.arctan2(dy, dx))
        slants = slants[pen_down[:-1]]  # only pen-down segments
        
        features["mean_slant"] = np.mean(slants) if len(slants) else 0
        features["std_slant"] = np.std(slants) if len(slants) else 0
        features["abs_mean_slant"] = np.mean(np.abs(slants)) if len(slants) else 0
    else:
        features["mean_slant"] = 0
        features["std_slant"] = 0
        features["abs_mean_slant"] = 0
        
        # --- Pen-up features ---
    pen_up_pairs = pen_up[:-1] & pen_up[1:]

    # Pen-up path length
    if np.any(pen_up_pairs):
        pu_dx = dx[pen_up_pairs]
        pu_dy = dy[pen_up_pairs]
        features["pen_up_path_length"] = np.sum(np.sqrt(pu_dx**2 + pu_dy**2))
    else:
        features["pen_up_path_length"] = 0

    # Pen-up speeds
    if len(speeds):
        speeds_up = speeds[pen_up_pairs[:len(speeds)]]
        features["median_pen_up_speed"] = np.median(speeds_up) if len(speeds_up) else 0
        features["p95_pen_up_speed"] = np.percentile(speeds_up, 95) if len(speeds_up) else 0
    else:
        features["median_pen_up_speed"] = 0
        features["p95_pen_up_speed"] = 0

    # Pen-up ratio
    total_path = features["path_length"]
    features["pen_up_ratio"] = (
        features["pen_up_path_length"] / total_path if total_path > 0 else 0
    )

    
    return features      
    
    