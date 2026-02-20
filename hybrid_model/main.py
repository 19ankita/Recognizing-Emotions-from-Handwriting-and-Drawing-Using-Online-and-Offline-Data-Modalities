from preprocessing.trajectory_processing import parse_whiteboard_xml, normalize_trajectory, resample_trajectory
from preprocessing.render_iam_images import render_xml_to_image
from training.train_reverse_model import run_train
from utils.sanity_check_reverse import run_sanity_check
from inference.generate_pseudo_trajectories_emothaw import generate_pseudo_trajectories
from utils.plot_emothaw_pseudo_trajectories import visualize_pseudo_trajectories
from features.extract_pseudo_features_emothaw import run_pseudo_feature_extraction
# from inference.run_pseudo import main as run_pseudo_main


import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# CONFIG
# ---------------------------
TRAJ_DIR = "data/raw/IAM_OnDB/trajectories"
IMG_DIR  = "data/raw/IAM_OnDB/images"
OUT_DIR  = "data/processed/IAM_OnDB/trajectories_npy"
META_CSV = "data/processed/IAM_OnDB/metadata.csv"

NUM_POINTS = 200

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

        # Render images from the online trajectories for IAM_OnDB
        render_xml_to_image(xml_path, img_path)

        # Parse trajectory
        traj = parse_whiteboard_xml(xml_path)
        if len(traj) < 10:
            continue

        # Normalize
        traj = normalize_trajectory(traj)

        # Resample
        traj = resample_trajectory(traj, NUM_POINTS)

        # Save the trajectories for IAM_OnDB
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

    # Train reverse model on IAM_onDB
    print("Training reverse model...")
    run_train()
    
    print("Sanity check for the reverse...")
    run_sanity_check()

    # Run pseudo generation from EMOTHAW
    print("Running pseudo generation...")
    generate_pseudo_trajectories()
    
    print("Visualizing pseudo trajectories...")
    visualize_pseudo_trajectories(
        task="cursive_writing",
        num_samples=2
    )

    # Generate EMOTHAW pseudo trajectories
    print("Extracting pseudo trajectory features...")
    run_pseudo_feature_extraction(
        tasks=("cursive_writing",),
        traj_root="data/processed/EMOTHAW/pseudo_trajectories",
        out_root="data/processed/EMOTHAW/pseudo_features",
    )

    # # 7. Extract trajectory features
    # print("Extracting trajectory features...")
    # feature_extraction_main()

if __name__ == "__main__":
    main()
