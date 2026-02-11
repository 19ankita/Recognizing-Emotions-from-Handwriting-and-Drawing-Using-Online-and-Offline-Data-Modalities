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


def main():
    traj_dir = "data/IAM_OnDB/trajectories"
    img_dir  = "data/IAM_OnDB/images"

    os.makedirs(img_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(traj_dir) if f.endswith(".xml")]
    print(f"Found {len(xml_files)} XML files")

    for i, fname in enumerate(xml_files):
        xml_path = os.path.join(traj_dir, fname)
        img_path = os.path.join(img_dir, fname.replace(".xml", ".png"))

        render_xml_to_image(xml_path, img_path)

        if (i + 1) % 100 == 0:
            print(f"Rendered {i + 1}/{len(xml_files)}")

    print("Image rendering completed")


if __name__ == "__main__":
    main()
