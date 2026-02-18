import argparse
import os
import sys
import re
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from src.svc_reader import read_all_svc_files
from src.run_feature_extractor import run_feature_extraction
from src.training_models import run_model, run_multioutput_model, run_separate_subscale_models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ===================== PATH SETUP  =====================
base_dir = os.path.dirname(os.path.abspath(__file__))

dataset_dir  = os.path.join(base_dir, "dataset")
labels_dir   = os.path.join(base_dir, "labels")
features_dir = os.path.join(base_dir, "features")
results_dir  = os.path.join(base_dir, "results")

# ===========================================================


def extract_global_user(id_str):

    """
    Map a file/sample ID string to a global user index across two collections.

    Input
    -----
    id_str : str
        Sample identifier containing a user marker of the form:
        - 'u<digits>' for Collection 1 users (1–45)
        - 'v<digits>' for Collection 2 users (1–84)
        Example: "u12_taskX" or "v03_house".

    Output
    ------
    user_id : int or None
        Global user ID:
        - If prefix is 'u': returns the extracted user number unchanged.
        - If prefix is 'v': returns extracted user number + 45 (offset to avoid overlap).
        Returns None if no pattern is found.
    """
    match = re.search(r"([uv])(\d+)", id_str)
    if not match:
        return None
    
    prefix, user = match.group(1), int(match.group(2))
    
    if prefix == "u":        # Collection 1
        return user
    elif prefix == "v":      # Collection 2
        return user + 45


def prepare_labels():

    """
    Load DASS subscale labels, convert collection-specific user IDs into a single
    global user index, compute TOTAL DASS, and save the merged label table.

    Input
    -----
    Reads:
      - labels/DASS_scores_clean.csv (semicolon-separated), expected to contain at least:
        ['user', 'depression', 'anxiety', 'stress'] in the first 45 rows for Collection 1,
        and the remaining rows for Collection 2 (user numbering restarts).

    Output
    ------
    labels_global : pandas.DataFrame
        Cleaned label table with globally unique user IDs and an added 'total' column.
        Also writes:
          - labels/DASS_scores_global.csv (comma-separated by default in pandas)

    Notes
    -----
    - Collection 2 users are offset by +45 to make user IDs unique.
    - TOTAL DASS is computed as depression + anxiety + stress.
    """
    os.makedirs(labels_dir, exist_ok=True)
    
    input_path = os.path.join(labels_dir, "DASS_scores_clean.csv")
    output_path = os.path.join(labels_dir, "DASS_scores_global.csv")

    labels = pd.read_csv(input_path, sep=";")
    print("LABEL COLUMNS:", labels.columns.tolist())
    
    # Split collections
    labels_c1 = labels.iloc[:45].copy()     # users 1–45
    labels_c2 = labels.iloc[45:].copy()     # users 1–84 again
    
    # Offset Collection 2 users
    labels_c2["user"] = labels_c2["user"] + 45
    
    # Combine
    labels_global = pd.concat([labels_c1, labels_c2], ignore_index=True)
    
    # Add total score
    labels_global.loc[:,"total"] = (
    labels_global["depression"] + labels_global["anxiety"] + labels_global["stress"]
)
       
    labels_global.to_csv(output_path, index=False)

    print(f"Cleaned DASS labels saved to {labels_global}")
    
    return labels_global


def main():

    """
    End-to-end experiment runner for handwriting-based DASS regression.

    Pipeline
    --------
    1) Parse CLI arguments to select tasks and experiment mode:
       - total     : predict TOTAL DASS
       - multi     : joint multi-output prediction (depression, anxiety, stress)
       - subscales : train separate models for each subscale
       - all       : run all of the above
    2) Load and prepare global DASS labels (prepare_labels()).
    3) For each task:
       - Read all .svc/.SVC files from dataset/<task>/ (read_all_svc_files)
       - Convert trajectories to a point-level DataFrame with an 'id'
       - Extract per-id feature vectors (run_feature_extraction)
       - Merge features with DASS labels via a global user ID mapping (extract_global_user)
    4) Train and evaluate multiple models (linear + regularized + tree-based),
       optionally with cross-validation, hyperparameter search, and SHAP explanations.
    5) Save a summary CSV of all experiment metrics to results/model_summary.csv.

    Input
    -----
    Command-line arguments:
      tasks (positional) : list of task folder names under dataset/
      --all-tasks        : run all tasks found in dataset/
      --mode             : {total, subscales, multi, all}
      --cv               : enable K-Fold CV reporting
      --search           : enable GridSearchCV/RandomizedSearchCV
      --cv-folds         : number of folds (default=5)
      --shap             : run SHAP analysis per trained model

    Output
    ------
    None
        Writes intermediate feature CSVs to features/ and a final model summary CSV:
        - features/<task>_features.csv
        - features/<task>_with_dass.csv
        - results/model_summary.csv
    """
    # --- Parse command line arguments ---
    parser = argparse.ArgumentParser(description="Run regression experiments on handwriting tasks")
    parser.add_argument("tasks", nargs="*", help="List of tasks (e.g., words cursive house)")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Run experiments on all available tasks inside dataset/ folder.")
    parser.add_argument("--mode", choices=["total", "subscales", "multi", "all"], default="all",
                        help="Experiment mode: total (TOTAL DASS), subscales (Depression/Anxiety/Stress separately), multi (multi-output), all (default).")
    parser.add_argument("--cv", action="store_true", help="Enable cross-validation reporting.")
    parser.add_argument("--search", action="store_true", help="Enable hyperparameter search (Grid/RandomizedSearchCV).")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of folds for cross-validation (default=5).")
    parser.add_argument("--shap", action="store_true", help="Run SHAP analysis for each model.")

    args = parser.parse_args()

    # Select tasks
    if args.all_tasks:
        tasks = [t for t in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, t))]
        print(f"\n Detected tasks automatically: {tasks}")
    else:
        tasks = args.tasks
        if not tasks:
            print("Please provide at least one task name or use --all-tasks")
            sys.exit(1)

    mode = args.mode
    
    # Load cleaned DASS labels 
    labels = prepare_labels()
    os.makedirs(features_dir, exist_ok=True)
    results = []
    
    # ===================== MODELS =====================

    models = []

    # Linear + Ridge WITH PCA
    for name, base_model in [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge())
    ]:
        models.append((
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95)),
                ("model", base_model)
            ]),
            name
        ))
        
    # Lasso / ElasticNet WITHOUT PCA
    for name, base_model in [
        ("Lasso Regression", Lasso()),
        ("Elastic Net", ElasticNet())
    ]:
        models.append((
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", base_model)
            ]),
            name
        ))
        
    # Tree models (NO scaling / PCA)
    models.extend([
        (RandomForestRegressor(random_state=42), "Random Forest"),
        (GradientBoostingRegressor(random_state=42), "Gradient Boosting")
    ])

    
    for task in tasks:
        input_dir = os.path.join(dataset_dir, task)
        output_csv = os.path.join(features_dir, f"{task}_features.csv")
        merged_csv = os.path.join(features_dir, f"{task}_with_dass.csv")
        
        if not os.path.exists(input_dir):
            print(f"Task folder not found: {input_dir}, skipping")
            continue
        
        print(f"Processing task: {task}")   
              
        # Load all .svc files as dictionary
        data = read_all_svc_files(input_dir)
        print(f"Loaded {len(data)} files for task '{task}")
        
        # Convert the dictionary to dataframe (with file name as id)
        columns = ["x", "y", "timestamp", "pen_status", "azimuth", "altitude", "pressure"]
        dfs = []
        for id, arr in data.items():
            df = pd.DataFrame(arr, columns=columns)
            df["id"] = id
            dfs.append(df)
        full_df = pd.concat(dfs, ignore_index=True)   
            
        # Feature engineering
        run_feature_extraction(
            full_df,
            output_csv 
        )
        
        # --- Merge with DASS labels ---
        features = pd.read_csv(output_csv)
        features["user"] = features["id"].apply(extract_global_user)
        
        merged = features.merge(labels, on="user", how="inner")
        merged.to_csv(merged_csv, index=False)
        
        # Run single-output models
        for model, model_name in models:
            if mode in ["total", "all"]:
                # 1. Single-output (TOTAL DASS)
                results.append(run_model(merged_csv, task, model, mode, model_name, target="total",
                                         do_cv=args.cv, do_search=args.search, cv_folds=args.cv_folds,
                                         do_shap=args.shap
                                         ))
            
            if mode in ["multi", "all"]:
                # 2. Multi-output (Depression, Anxiety, Stress simultaneously)
                results.extend(run_multioutput_model(merged_csv, task, model, mode, model_name,
                               do_cv=args.cv, do_search=args.search, cv_folds=args.cv_folds,
                               do_shap=args.shap
                               ))

            if mode in ["subscales", "all"]:
                # 3. Separate models for each subscale
                results.extend(run_separate_subscale_models(merged_csv, task, model, mode, model_name,
                                                            do_cv=args.cv, do_search=args.search, cv_folds=args.cv_folds,
                                                            do_shap=args.shap))
                   
    
    if results:
        os.makedirs(results_dir, exist_ok=True)
        summary_csv = os.path.join(results_dir, "model_summary.csv")
        pd.DataFrame(results).to_csv(summary_csv, index=False)
        print(f"\n Summary of regression results saved to {summary_csv}")
 
               
if __name__ == "__main__":
        main()    
        