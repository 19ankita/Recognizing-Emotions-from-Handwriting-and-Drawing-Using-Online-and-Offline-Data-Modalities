import xml.etree.ElementTree as ET
import numpy as np

def parse_whiteboard_xml(xml_path):
    """
    Parses a WhiteboardCaptureSession XML file into a trajectory array.

    Returns:
        traj: numpy array of shape (N, 3) -> [x, y, time]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    points = []

    for stroke in root.iter("Stroke"):
        for point in stroke.iter("Point"):
            try:
                x = float(point.attrib["x"])
                y = float(point.attrib["y"])
                t = float(point.attrib["time"])
            except KeyError:
                continue

            points.append([x, y, t])

    return np.array(points)
