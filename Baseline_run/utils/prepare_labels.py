import pandas as pd
import os

def prepare_labels(input_path="DASS_scores.xls", output_path="labels/DASS_scores_clean.csv"):
    
    """
    Clean the raw DASS Excel file and compute the Total DASS score.
    
    Expected columns in input Excel:
        user, depression, anxiety, stress
    Output CSV includes:
        user, depression, anxiety, stress, total
    """
    # Load Excel file
    df = pd.read_excel("E:/2nd_thesis/Experiment/DASS_scores.xls")
    
    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Ensure essential columns exist
    expected_cols = ["user", "depression", "anxiety", "stress"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col} in {input_path}")

    # Compute total DASS score
    df["total"] = df["depression"] + df["anxiety"] + df["stress"]
    
    # Drop rows with any missing values (optional)
    df = df.dropna(subset=expected_cols)
    
    # Create output folder if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save cleaned CSV
    df.to_csv(output_path, index=False)
    print(f"Cleaned DASS scores saved to: {output_path}")

    return df

# Example usage
if __name__ == "__main__":
    clean_df = prepare_labels()
    print(clean_df.head())
