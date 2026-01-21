import os
import numpy as np
import pandas as pd

from src.utils.parse_xml import parse_whiteboard_xml, normalize_trajectory, resample_trajectory, render_xml_to_image

# ---------------------------
# CONFIG
# ---------------------------
TRAJ_DIR = "data/IAM_OnDB/trajectories"
IMG_DIR  = "data/IAM_OnDB/images"
OUT_DIR  = "data/IAM_OnDB/processed/trajectories_npy"
META_CSV = "data/IAM_OnDB/processed/metadata.csv"

NUM_POINTS = 200

# ---------------------------
# MAIN PIPELINE
# ---------------------------
def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    records = []

    xml_files = [f for f in os.listdir(TRAJ_DIR) if f.endswith(".xml")]
    print(f"Found {len(xml_files)} XML files")

    for i, fname in enumerate(xml_files):
        xml_path = os.path.join(TRAJ_DIR, fname)
        img_path = os.path.join(IMG_DIR, fname.replace(".xml", ".png"))
        out_path = os.path.join(OUT_DIR, fname.replace(".xml", ".npy"))

        # Render image
        render_xml_to_image(xml_path, img_path)

        # Parse trajectory
        traj = parse_whiteboard_xml(xml_path)
        if len(traj) < 10:
            continue

        # Normalize
        traj = normalize_trajectory(traj)

        # Resample
        traj = resample_trajectory(traj, NUM_POINTS)

        # 5️Save
        np.save(out_path, traj)

        records.append({
            "id": fname.replace(".xml", ""),
            "xml": xml_path,
            "image": img_path,
            "trajectory": out_path
        })

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(xml_files)}")

    pd.DataFrame(records).to_csv(META_CSV, index=False)
    print("Pipeline completed successfully")

if __name__ == "__main__":
    main()
