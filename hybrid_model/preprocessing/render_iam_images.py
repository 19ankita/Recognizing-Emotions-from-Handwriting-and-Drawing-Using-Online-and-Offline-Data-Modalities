import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def render_xml_to_image(xml_path, out_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    plt.figure(figsize=(6, 3))
    ax = plt.gca()
    plotted = False

    for stroke in root.iter("Stroke"):
        xs, ys = [], []

        for point in stroke.iter("Point"):
            if "x" not in point.attrib or "y" not in point.attrib:
                continue
            xs.append(float(point.attrib["x"]))
            ys.append(float(point.attrib["y"]))

        if len(xs) > 1:
            ax.plot(xs, ys, color="black", linewidth=2)
            plotted = True

    if not plotted:
        plt.close()
        return

    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()          
    ax.axis("off")

    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()