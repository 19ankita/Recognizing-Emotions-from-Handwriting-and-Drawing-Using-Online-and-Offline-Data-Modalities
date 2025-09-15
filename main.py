import os
import sys
import re
import pandas as pd

from src.svc_reader import read_all_svc_files
from src.run_feature_extractor import run_feature_extraction
from src.training_regression import run_single_output_regression


def extract_user_number(x):
    """Extract user number from feature id (e.g. u00025s00001_hw00003 → 25)."""
    match = re.search(r"u(\d+)", x)
    return int(match.group(1)) if match else None

def prepare_labels():
    """Clean the DASS Excel file and save to labels/DASS_scores_clean.csv"""
    labels_dir = "labels"
    os.makedirs(labels_dir, exist_ok=True)
    
    excel_file = "DASS_scores.xls"
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
    
    # Check the comman-line argument
    if len(sys.argv) < 2:
        print("Please provide a task name (e.g., house, clock, pentagon).")
        print("Usage: python main.py <task_name>")
        sys.exit(1)
    
    # Collect all tasks from the command line
    tasks = sys.argv[1:]
    
    # Load cleaned DASS labels 
    labels = prepare_labels()
    
    for task in tasks:
        input_dir = os.path.join("dataset", task)
        output_csv = os.path.join("features", f"{task}_features.csv")
        merged_csv = os.path.join("features", f"{task}_with_dass.csv")
        
        if not os.path.exists(input_dir):
            print(f"Task folder not found: {input_dir}, skipping")
            continue
        
        print(f"Processing task: {task}")   
              
        # Load all .svc files as dictionary
        data = read_all_svc_files("dataset/words")
        print(f"Loaded {len(data)} files")
        
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
        print("Sample rows:\n", full_df.head(10))      
            
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
            
        if task == "words":
            run_single_output_regression(merged_csv)
    
    
if __name__ == "__main__":
    main()    
    