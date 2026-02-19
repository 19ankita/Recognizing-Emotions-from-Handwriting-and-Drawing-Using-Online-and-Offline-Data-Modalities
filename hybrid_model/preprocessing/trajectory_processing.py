import xml.etree.ElementTree as ET
import numpy as np
import os

def parse_whiteboard_xml(xml_path):
    points = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for stroke in root.iter("Stroke"):
        for point in stroke.iter("Point"):
            x = float(point.attrib["x"])
            y = float(point.attrib["y"])
            t = float(point.attrib["time"])
            points.append([x, y, t])

    return np.array(points)


def normalize_trajectory(traj):
    traj = traj.copy()

    traj[:, 0] = (traj[:, 0] - traj[:, 0].min()) / (np.ptp(traj[:, 0]) + 1e-8)
    traj[:, 1] = (traj[:, 1] - traj[:, 1].min()) / (np.ptp(traj[:, 1]) + 1e-8)

    traj[:, 2] = traj[:, 2] - traj[:, 2][0]

    return traj


def resample_trajectory(traj, num_points=200):
    idx = np.linspace(0, len(traj) - 1, num_points)
    idx = idx.astype(int)
    return traj[idx]
