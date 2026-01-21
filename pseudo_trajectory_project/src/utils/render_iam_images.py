import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def render_xml_to_image(xml_path, out_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    plt.figure(figsize=(6, 3))

    for trace in root.iter("trace"):
        if trace.text is None:
            continue

        points = trace.text.strip().split(",")
        xs, ys = [], []

        for p in points:
            vals = p.strip().split()
            if len(vals) < 2:
                continue
            xs.append(float(vals[0]))
            ys.append(-float(vals[1]))  # invert Y for correct orientation

        if len(xs) > 1:
            plt.plot(xs, ys, linewidth=2, color="black")

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    traj_dir = r"data/IAM_OnDB/trajectories"
    img_dir  = r"data/IAM_OnDB/images"

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
