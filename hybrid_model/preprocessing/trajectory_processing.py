import xml.etree.ElementTree as ET
import numpy as np


def parse_whiteboard_xml(xml_path):
    points = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    stroke_id = 0
    for stroke in root.iter("Stroke"):
        for point in stroke.iter("Point"):
            x = float(point.attrib["x"])
            y = float(point.attrib["y"])
            t = float(point.attrib["time"])
            points.append([x, y, t, stroke_id])
        stroke_id += 1

    return np.array(points, dtype=np.float32)  # [N,4] = x,y,t,stroke_id


def normalize_trajectory(traj):
    traj = traj.copy()

    traj[:, 0] = (traj[:, 0] - traj[:, 0].min()) / (np.ptp(traj[:, 0]) + 1e-8)
    traj[:, 1] = (traj[:, 1] - traj[:, 1].min()) / (np.ptp(traj[:, 1]) + 1e-8)
    traj[:, 2] = traj[:, 2] - traj[:, 2][0]

    return traj


def resample_trajectory(traj, num_points=200):
    """
    Returns:
        seq: [T,3] where columns are (x, y, pen_down)
             pen_down = 0 at the first point of each new stroke (except first stroke),
             otherwise pen_down = 1.
    Input:
        traj: [N,4] = (x, y, t, stroke_id) normalized in [0,1] for x,y
    """
    if traj.shape[1] != 4:
        raise ValueError(f"Expected traj shape [N,4], got {traj.shape}")

    sids = traj[:, 3].astype(int)
    unique = np.unique(sids)

    # If only one stroke, fall back to simple resample + all pen down
    if len(unique) == 1:
        idx = np.linspace(0, len(traj) - 1, num_points).astype(int)
        sampled = traj[idx]
        pen = np.ones((len(sampled), 1), dtype=np.float32)
        return np.concatenate([sampled[:, :2], pen], axis=1).astype(np.float32)

    # Allocate points per stroke proportional to stroke length
    lengths = np.array([(sids == s).sum() for s in unique], dtype=np.float32)
    alloc = np.maximum(2, np.round(num_points * lengths / lengths.sum()).astype(int))

    # Adjust allocation to hit num_points exactly
    while alloc.sum() > num_points:
        alloc[np.argmax(alloc)] -= 1
    while alloc.sum() < num_points:
        alloc[np.argmax(lengths)] += 1

    out = []
    for j, (s, k) in enumerate(zip(unique, alloc)):
        pts = traj[sids == s]  # [Ns,4]

        if len(pts) <= k:
            sampled = pts
        else:
            idx = np.linspace(0, len(pts) - 1, k).astype(int)
            sampled = pts[idx]

        pen = np.ones((len(sampled), 1), dtype=np.float32)

        # Mark the first point of every NEW stroke (except the first stroke) as pen-up transition
        if j > 0 and len(sampled) > 0:
            pen[0, 0] = 0.0

        out.append(np.concatenate([sampled[:, :2], pen], axis=1))  # [k,3]

    seq = np.concatenate(out, axis=0).astype(np.float32)  # [T,3]

    # Safety: if rounding made it slightly off, crop/pad
    if len(seq) > num_points:
        seq = seq[:num_points]
    elif len(seq) < num_points:
        pad = np.repeat(seq[-1][None, :], num_points - len(seq), axis=0)
        pad[:, 2] = 0.0  # padded points are pen-up
        seq = np.concatenate([seq, pad], axis=0)

    return seq