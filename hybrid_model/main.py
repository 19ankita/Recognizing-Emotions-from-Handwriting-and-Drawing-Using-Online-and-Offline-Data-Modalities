from preprocessing.trajectory_processing import parse_whiteboard_xml, normalize_trajectory, resample_trajectory
from preprocessing.render_iam_images import render_xml_to_image
from training.train_reverse_model import run_train
from utils.sanity_check_reverse import run_sanity_check
from inference.generate_pseudo_trajectories_emothaw import generate_pseudo_trajectories
from utils.plot_emothaw_pseudo_trajectories import visualize_pseudo_trajectories
from features.extract_pseudo_features_emothaw import run_pseudo_feature_extraction
from training.run_regression import run_regression


import os
import numpy as np
import pandas as pd
import re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# CONFIG
# ---------------------------
TRAJ_DIR = "data/raw/IAM_OnDB/trajectories"
IMG_DIR  = "data/raw/IAM_OnDB/images"
OUT_DIR  = "data/processed/IAM_OnDB/trajectories_npy"
META_CSV = "data/processed/IAM_OnDB/metadata.csv"
LABELS_CSV = "labels/DASS_scores_global.csv"

features_csv = "data/processed/EMOTHAW/pseudo_features/cursive_writing_pseudo_features.csv"
merged_csv   = "data/processed/EMOTHAW/pseudo_features/cursive_writing_pseudo_with_labels.csv"

NUM_POINTS = 200

def extract_global_user(id_str: str):
    """
    Extract global user index from sample id.
    u01..u45 -> 1..45
    v01..v84 -> 46..129 (offset +45)
    """
    if not isinstance(id_str, str):
        return None

    match = re.search(r"([uv])(\d+)", id_str)
    if not match:
        return None

    prefix, user = match.group(1), int(match.group(2))

    if prefix == "u":
        return user
    if prefix == "v":
        return user + 45
    return None

def merge_pseudo_features_with_labels(
    features_csv: str,
    labels_csv: str,
    merged_csv: str,
):
    # Load
    features = pd.read_csv(features_csv, sep=None, engine="python")
    labels = pd.read_csv(labels_csv, sep=None, engine="python")

    # Ensure features have user
    features["user"] = features["id"].apply(extract_global_user)

    # Make sure labels have "user" too
    # If labels already have a numeric user column -> keep it.
    # If labels has an id-like column (e.g., "id"), map it.
    if "user" not in labels.columns:
        # common alternatives people use:
        for candidate in ["id", "subject", "participant", "user_id"]:
            if candidate in labels.columns:
                labels["user"] = labels[candidate].apply(extract_global_user)
                break

    # Keep only rows with valid user
    features = features.dropna(subset=["user"])
    labels = labels.dropna(subset=["user"])

    features["user"] = features["user"].astype(int)
    labels["user"] = labels["user"].astype(int)

    # Merge
    merged = features.merge(labels, on="user", how="inner")

    # Save
    os.makedirs(os.path.dirname(merged_csv), exist_ok=True)
    merged.to_csv(merged_csv, index=False)
    print(f"[INFO] Saved merged pseudo+labels to {merged_csv} ({len(merged)} rows)")

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

    # Run pseudo trajectories generation from EMOTHAW
    print("Running pseudo generation...")
    generate_pseudo_trajectories()
    
    print("Visualizing pseudo trajectories...")
    visualize_pseudo_trajectories(
        task="cursive_writing",
        num_samples=2
    )

    # Generate EMOTHAW pseudo features
    print("Extracting pseudo trajectory features...")
    run_pseudo_feature_extraction(
        tasks=("cursive_writing",),
        traj_root="data/processed/EMOTHAW/pseudo_trajectories",
        out_root="data/processed/EMOTHAW/pseudo_features",
    )
    
    # Merge the pseudo features with the DASS labels
    print("Merging the EMOTHAW pseudo features with the DASS labels...")
    merge_pseudo_features_with_labels(
    features_csv=features_csv,
    labels_csv=LABELS_CSV,   # "labels/DASS_scores_global.csv"
    merged_csv=merged_csv
    )

    print("Regression run...")
    run_regression()
    
    print("Regression run...")
    run_regression(
        merged_csv=merged_csv,
        task="cursive_writing",
        out_root="results/pseudo",
        targets=("stress", "anxiety", "depression", "total")
)

if __name__ == "__main__":
    main()
