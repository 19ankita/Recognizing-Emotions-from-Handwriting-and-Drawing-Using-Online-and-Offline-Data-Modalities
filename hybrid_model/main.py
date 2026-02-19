# ============================================
# HYBRID EMOTION FULL PIPELINE
# ============================================

from preprocessing.parse_xml import main as parse_xml_main
from preprocessing.preprocessing import main as preprocessing_main
from preprocessing.render_iam_images import main as render_images_main

from training.train_reverse_model import main as train_reverse_main

from inference.run_pseudo import main as run_pseudo_main
from inference.generate_pseudo_trajectories_emothaw import main as generate_pseudo_main

from features.trajectory_features import main as feature_extraction_main


def main():

    print("\n==============================")
    print(" HYBRID EMOTION PIPELINE START")
    print("==============================\n")

    # 1. Parse IAM XML
    print("Step 1: Parsing IAM XML...")
    parse_xml_main()

    # 2. Preprocess trajectories
    print("Step 2: Preprocessing trajectories...")
    preprocessing_main()

    # 3. Render IAM images
    print("Step 3: Rendering IAM images...")
    render_images_main()

    # 4. Train reverse model
    print("Step 4: Training reverse model...")
    train_reverse_main()

    # 5. Run pseudo generation
    print("Step 5: Running pseudo generation...")
    run_pseudo_main()

    # 6. Generate EMOTHAW pseudo trajectories
    print("Step 6: Generating pseudo trajectories for EMOTHAW...")
    generate_pseudo_main()

    # 7. Extract trajectory features
    print("Step 7: Extracting trajectory features...")
    feature_extraction_main()

    print("\n==============================")
    print(" PIPELINE COMPLETED SUCCESSFULLY ")
    print("==============================\n")


if __name__ == "__main__":
    main()
