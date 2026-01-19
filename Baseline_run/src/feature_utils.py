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

def acceleration(speeds, t):
    dt = np.diff(t[1:])   
    dv = np.diff(speeds)
    valid = dt > 0
    return dv[valid] / dt[valid]


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
    df must have columns: 
    ['x','y','timestamp','pen_status','azimuth','altitude','pressure','id']
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
    features["path_length"] = path_length(x, y)
    acc = acceleration(speeds, t)
    features["median_acceleration"] = np.median(acc) if len(acc) else 0

    features["median_speed"] = np.median(speeds_down) if len(speeds_down) else 0
    speeds_down = speeds[pen_down[1:]]
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
    
    return features      
    
    