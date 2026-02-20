import argparse

# -------------------------------
# IMPORT PIPELINE MODULES
# -------------------------------

from src.utils.parse_xml import main as parse_iam_xml
from src.utils.preprocessing import main as preprocess_trajectories
from src.utils.render_iam_images import main as render_images

from train_reverse_model import main as train_reverse
from run_pseudo import main as generate_pseudo

from src.features.trajectory_features import main as extract_features


def run_full_pipeline():

    print("\n==============================")
    print("STEP 1: Parsing IAM XML files")
    print("==============================")
    parse_iam_xml()

    print("\n==============================")
    print("STEP 2: Preprocessing trajectories")
    print("==============================")
    preprocess_trajectories()

    print("\n==============================")
    print("STEP 3: Rendering images")
    print("==============================")
    render_images()

    print("\n==============================")
    print("STEP 4: Training reverse model")
    print("==============================")
    train_reverse()

    print("\n==============================")
    print("STEP 5: Generating pseudo trajectories")
    print("==============================")
    generate_pseudo()

    print("\n==============================")
    print("STEP 6: Extracting trajectory features")
    print("==============================")
    extract_features()

    print("\n🎉 Pipeline completed successfully!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Hybrid Model Centralized Pipeline"
    )

    parser.add_argument("--all", action="store_true",
                        help="Run the full pipeline")

    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--pseudo_only", action="store_true")

    args = parser.parse_args()

    if args.all:
        run_full_pipeline()

    elif args.train_only:
        train_reverse()

    elif args.pseudo_only:
        generate_pseudo()

    else:
        print("Please specify a mode: --all | --train_only | --pseudo_only")
