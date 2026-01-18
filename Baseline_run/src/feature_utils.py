import numpy as np
from scipy.spatial import ConvexHull

# ---------- Utility functions ----------
def euclidean_dist(x1, y1, x2,  y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def path_length(x, y):
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

def straightness(x, y, pen_down):
    """
    Straightness computed on pen-down trajectory only.
    """
    x_pd = x[pen_down]
    y_pd = y[pen_down]

    if len(x_pd) < 2:
        return 0.0

    chord = euclidean_dist(x_pd[0], y_pd[0], x_pd[-1], y_pd[-1])
    length = np.sum(np.sqrt(np.diff(x_pd)**2 + np.diff(y_pd)**2))

    return chord / length if length > 0 else 0.0


def stop_ratio(speeds):
    if len(speeds) == 0:
        return 0
    
    eps = 0.1 * np.median(speeds[speeds > 0]) if np.any(speeds > 0) else 0
    return np.sum(speeds < eps) / len(speeds) if eps > 0 else 0

def curvature(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    theta = np.arctan2(dy, dx)

    dtheta = np.diff(theta)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # wrap

    dr = np.sqrt(dx**2 + dy**2)
    s = 0.5 * (dr[1:] + dr[:-1])

    valid = s > 0
    return dtheta[valid] / s[valid]

def segment_lines(x, y, pen_down, y_gap_factor=0.02):
    """
    Segments pen-down points into lines using vertical gaps.
    """
    xd = x[pen_down]
    yd = y[pen_down]

    if len(yd) < 10:
        return []

    # Sort by y (top to bottom)
    idx = np.argsort(yd)
    xd, yd = xd[idx], yd[idx]

    # Vertical gaps
    dy = np.diff(yd)
    H = yd.max() - yd.min()
    gap_thresh = y_gap_factor * H

    lines = []
    start = 0
    for i in range(len(dy)):
        if dy[i] > gap_thresh:
            lines.append((xd[start:i+1], yd[start:i+1]))
            start = i + 1
    lines.append((xd[start:], yd[start:]))

    return [line for line in lines if len(line[0]) > 5]

def slant_features(lines, phi_max=30):
    slants = []

    for x_l, y_l in lines:
        if len(x_l) < 3:
            continue

        dx = np.diff(x_l)
        dy = np.diff(y_l)

        # Signed slant: deviation from vertical
        phi = np.degrees(np.arctan2(dx, dy))  # note order: (dx, dy)

        # Keep near-vertical strokes only
        phi = phi[np.abs(phi) <= phi_max]

        if len(phi):
            slants.extend(phi)

    if len(slants) == 0:
        return 0, 0

    slants = np.array(slants)
    slant_deg = np.median(slants)
    dispersion = np.percentile(slants, 75) - np.percentile(slants, 25)

    return slant_deg, dispersion

def word_spacing_features(lines, eta=0.01):
    gaps = []

    for x_l, y_l in lines:
        if len(x_l) < 5:
            continue

        # Sort points by x
        idx = np.argsort(x_l)
        x_sorted = x_l[idx]

        # Identify stroke segments via large x jumps
        dx = np.diff(x_sorted)

        W_l = x_sorted.max() - x_sorted.min()
        if W_l <= 0:
            continue

        # Gaps between consecutive segments
        for g in dx:
            if g > eta * W_l:
                gaps.append(g / W_l)

    if len(gaps) == 0:
        return 0, 0, 0, 0

    gaps = np.array(gaps)
    return (
        np.mean(gaps),
        np.median(gaps),
        np.percentile(gaps, 95),
        len(gaps)
    )


def line_spacing_features(lines, H):
    """
    Computes normalized line spacing statistics.

    Parameters
    ----------
    lines : list of tuples
        Each element is (x_l, y_l) for one handwriting line.
    H : float
        Height of the global text bounding box.

    Returns
    -------
    mean_spacing : float
        Mean normalized line spacing.
    std_spacing : float
        Standard deviation of normalized line spacing.
    """

    if len(lines) < 2 or H <= 0:
        return 0.0, 0.0

    # Robust vertical location per line (median y)
    y_centers = [np.median(y_l) for _, y_l in lines]

    # Sort top to bottom
    y_centers = np.sort(y_centers)

    # Vertical distances between consecutive baselines
    spacings = np.diff(y_centers) / H

    if len(spacings) == 0:
        return 0.0, 0.0

    return np.mean(spacings), np.std(spacings)


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
    dt_full = np.diff(t, prepend=t[0])
    
    features["F1_in_air_time"] = np.sum(dt_full[pen_up])
    features["F2_on_paper_time"] = np.sum(dt_full[pen_down])
    features["F3_total_time"] = t[-1] - t[0]
    features["F4_stroke_count"] = np.sum(np.diff(status) == 1) + 1
    
    
    # Duty cycle
    features["duty_cycle"] = (
        features["F2_on_paper_time"] / features["F3_total_time"]
        if features["F3_total_time"] > 0 else 0
    )
    
    
    # --- Kinematic ---
    dx = np.diff(x)
    dy = np.diff(y)
    dt_diff = np.diff(t)

    valid = dt_diff > 0
    if np.any(valid):
        vx = dx[valid] / dt_diff[valid]
        vy = dy[valid] / dt_diff[valid]
        speeds = np.sqrt(vx**2 + vy**2)

        pen_down_valid = pen_down[1:][valid]
        speeds_down = speeds[pen_down_valid]
    else:
        speeds_down = np.array([])

    features["path_length"] = (
        path_length(x[pen_down], y[pen_down]) if np.any(pen_down) else 0.0
    )

    features["median_speed"] = np.median(speeds_down) if len(speeds_down) else 0

    features["p95_speed"] = np.percentile(speeds_down, 95) if len(speeds_down) else 0
    features["stop_ratio"] = stop_ratio(speeds_down)  
    
    
    # --- Geometric ---      
    features["straightness"] = straightness(x, y, pen_down)
    dx, dy = np.diff(x), np.diff(y)
    
    if len(dx) > 0:
        thetas = np.arctan2(dy, dx)
        C, S = np.mean(np.cos(thetas)), np.mean(np.sin(thetas))
        features["dominant_angle"] = np.degrees(np.arctan2(S, C))
        features["direction_concentration"] = (C**2 + S**2) / len(thetas)
    else:
        features["dominant_angle"] = 0
        features["direction_concentration"] = 0        
     
    # --- Curvature ---     
    kappa = curvature(x, y)
    features["median_curvature"] = np.median(np.abs(kappa)) if len(kappa) else 0
    
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
    
    # --- Baseline angle & waviness ---
    lines = segment_lines(x, y, pen_down)

    angles = []
    waviness = []

    for x_l, y_l in lines:
        if len(x_l) < 5:
            continue

        # Least-squares line fit
        a_l, b_l = np.polyfit(x_l, y_l, 1)

        # Baseline angle (degrees)
        theta_l = np.degrees(np.arctan(a_l))
        angles.append(theta_l)

        # Vertical residuals
        residuals = y_l - (a_l * x_l + b_l)
        wav_vert = np.sqrt(np.mean(residuals**2))

        # Orthogonal waviness
        wav_orth = wav_vert / np.sqrt(1 + a_l**2)
        waviness.append(wav_orth)

    # Document-level summaries
    features["baseline_angle_mean"] = np.mean(angles) if angles else 0
    features["baseline_waviness_mean"] = np.mean(waviness) if waviness else 0

    # --- Left / Right margins ---
    if np.any(pen_down):
        Xmin = x[pen_down].min()
        Xmax = x[pen_down].max()
        W = Xmax - Xmin
    else:
        W = 0

    mL, mR = [], []

    if W > 0:
        for x_l, y_l in lines:
            if len(x_l) < 5:
                continue

            xmin_l = x_l.min()
            xmax_l = x_l.max()

            # Raw margins
            mL_l = xmin_l - Xmin
            mR_l = Xmax - xmax_l

            # Normalized margins
            mL.append(mL_l / W)
            mR.append(mR_l / W)

    # Aggregate statistics
    features["left_margin_mean"] = np.mean(mL) if mL else 0
    features["left_margin_std"]  = np.std(mL)  if mL else 0
    features["right_margin_mean"] = np.mean(mR) if mR else 0
    features["right_margin_std"]  = np.std(mR)  if mR else 0

    # --- Slant & dispersion ---
    slant_deg, slant_disp = slant_features(lines)

    features["slant_deg"] = slant_deg
    features["slant_dispersion"] = slant_disp

    # --- Word spacing ---
    ws_mean, ws_median, ws_p95, ws_count = word_spacing_features(lines)

    features["word_spacing_mean"] = ws_mean
    features["word_spacing_median"] = ws_median
    features["word_spacing_p95"] = ws_p95
    features["word_spacing_count"] = ws_count

    # --- Line spacing ---
    ls_mean, ls_std = line_spacing_features(lines, H)

    features["line_spacing_mean"] = ls_mean
    features["line_spacing_std"] = ls_std


    # --- Ink density ---
    area = 0  
    try:
        hull = ConvexHull(np.vstack((x[pen_down], y[pen_down])).T)
        area = hull.volume
    except: 
        area = 0
    features["ink_density"] = features["path_length"] / area if area > 0 else 0  
    
  # ================= Pen-up (In-air) features =================

    # Segment pen-up durations
    penup_durations = []
    current = 0.0

    for d, up in zip(dt_full, pen_up):
        if up:
            current += d
        else:
            if current > 0:
                penup_durations.append(current)
                current = 0.0

    if current > 0:
        penup_durations.append(current)

    penup_durations = np.array(penup_durations)

    features["penup_time_total"] = np.sum(dt_full[pen_up])
    features["penup_time_ratio"] = (
        features["penup_time_total"] / features["F3_total_time"]
        if features["F3_total_time"] > 0 else 0
    )

    features["penup_segment_count"] = len(penup_durations)
    features["penup_mean_duration"] = np.mean(penup_durations) if len(penup_durations) else 0
    features["penup_max_duration"] = np.max(penup_durations) if len(penup_durations) else 0
    features["penup_std_duration"] = np.std(penup_durations) if len(penup_durations) else 0

    features["stroke_pause_ratio"] = (
    features["penup_segment_count"] / features["F4_stroke_count"]
    if features["F4_stroke_count"] > 0 else 0
    )

    return features      
    
    