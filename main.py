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


def extract_user_number(x):
    """Extract user number from feature id (e.g. u00025s00001_hw00003 → 25)."""
    match = re.search(r"u(\d+)", x)
    return int(match.group(1)) if match else None

def prepare_labels():
    """Clean the DASS Excel file and save to labels/DASS_scores_clean.csv"""
    labels_dir = "labels"
    os.makedirs(labels_dir, exist_ok=True)
    
    output_file = os.path.join(labels_dir, "DASS_scores_clean.csv")
    
    if os.path.exists(output_file):
        print("Using the existing cleaned DASS labels..")
        return pd.read_csv(output_file)
    
    print("Preparing DASS labels...")
    
    dass = pd.read_excel("DASS_scores.xls",engine="xlrd")
    
    # Keep relevant columns
    dass_clean = dass.loc[:, ["File Number user", "depression", "anxiety", "stress"]].copy()
    dass_clean.rename(columns={"File Number user": "user"}, inplace=True)
    
    # Add total score
    dass_clean.loc[:,"total"] = (
    dass_clean["depression"] + dass_clean["anxiety"] + dass_clean["stress"]
)
    
    # Save clean labels
    dass_clean.to_csv(output_file, index=False)
    print(f"Cleaned DASS labels saved to {output_file}")
    
    return dass_clean


def main():

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
        tasks = [t for t in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", t))]
        print(f"\nDetected tasks automatically: {tasks}")
    else:
        tasks = args.tasks
        if not tasks:
            print("Please provide at least one task name or use --all-tasks")
            sys.exit(1)

    mode = args.mode
    
    # Load cleaned DASS labels 
    labels = prepare_labels()
    results = []
    
    # Define models
    linear_models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge(alpha=1.0)),
        ("Lasso Regression", Lasso(alpha=0.1)),
        ("Elastic Net", ElasticNet(alpha=0.1, l1_ratio=0.5))
    ]

    ensemble_models = [
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]
    
    # Apply scaling + PCA ONLY to linear models
    models = []

    for name, base_model in linear_models:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),  # keep 95% variance
            ("model", base_model)
        ])
        models.append((pipeline, name))

    # Ensemble models WITHOUT scaling/PCA
    for name, model in ensemble_models:
        models.append((model, name))
    
    for task in tasks:
        input_dir = os.path.join("dataset", task)
        output_csv = os.path.join("features", f"{task}_features.csv")
        merged_csv = os.path.join("features", f"{task}_with_dass.csv")
        
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
        
        print("Shape of DataFrame:", full_df.shape)
        print("Columns:", full_df.columns.tolist())
        print("Sample rows:\n", full_df.head(5))      
            
        # Feature engineering
        print("Extracting features..")
        run_feature_extraction(
            full_df,
            output_csv 
        )
        
        # --- Merge with DASS labels ---
        features = pd.read_csv(output_csv)
        features["user"] = features["id"].apply(extract_user_number)
        
        merged = features.merge(labels, on="user")
        merged.to_csv(merged_csv, index=False)
        
        print(f"Features merged with DASS scores saved to {merged_csv}")
        
        # Run single-output models
        for model, model_name in models:
            if mode in ["total", "all"]:
                # 1. Single-output (TOTAL DASS)
                results.append(run_model(merged_csv, task, model, model_name, target="total",
                                         do_cv=args.cv, do_search=args.search, cv_folds=args.cv_folds,
                                         do_shap=args.shap
                                         ))
            
            if mode in ["subscales", "all"]:
                # 2. Multi-output (Depression, Anxiety, Stress simultaneously)
                results.extend(run_multioutput_model(merged_csv, task, model, model_name,
                               do_cv=args.cv, do_search=args.search, cv_folds=args.cv_folds,
                               do_shap=args.shap
                               ))

            if mode in ["multi", "all"]:
                # 3. Separate models for each subscale
                results.extend(run_separate_subscale_models(merged_csv, task, model, model_name,
                                                            do_cv=args.cv, do_search=args.search, cv_folds=args.cv_folds,
                                                            do_shap=args.shap))
                   
    
    if results:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        summary_csv = os.path.join(results_dir, "model_summary.csv")  
               
    if __name__ == "__main__":
        main()    
        