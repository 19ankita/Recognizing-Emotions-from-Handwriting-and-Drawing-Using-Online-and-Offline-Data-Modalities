import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def render_xml_to_image(xml_path, out_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    plt.figure(figsize=(6, 3))
    plotted = False

    # Iterate over all strokes
    for stroke in root.iter("Stroke"):
        xs, ys = [], []

        for point in stroke.iter("Point"):
            try:
                x = float(point.attrib["x"])
                y = float(point.attrib["y"])
            except KeyError:
                continue

            xs.append(x)
            ys.append(-y)  # invert y-axis for image coordinates

        if len(xs) > 1:
            plt.plot(xs, ys, color="black", linewidth=2)
            plotted = True

    if not plotted:
        plt.close()
        return

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


