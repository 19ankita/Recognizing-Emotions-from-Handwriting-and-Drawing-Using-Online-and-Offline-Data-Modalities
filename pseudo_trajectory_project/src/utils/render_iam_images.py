import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def render_xml_to_image(xml_path, out_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    plt.figure(figsize=(6, 3))

    plotted = False  # track if anything is drawn

    for trace in root.iter("trace"):
        if trace.text is None:
            continue

        raw_points = trace.text.strip().replace("\n", " ").split(",")

        xs, ys = [], []

        for pt in raw_points:
            vals = pt.strip().split()
            if len(vals) < 2:
                continue

            try:
                x = float(vals[0])
                y = float(vals[1])
            except ValueError:
                continue

            xs.append(x)
            ys.append(-y)  # invert Y for image coordinates

        if len(xs) > 1:
            plt.plot(xs, ys, color="black", linewidth=2)
            plotted = True

    if not plotted:
        plt.close()
        return  # skip saving empty images

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
