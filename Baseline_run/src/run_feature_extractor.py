
import os
import pandas as pd
from .feature_utils import extract_features

def run_feature_extraction(full_df, output_csv):
    """
    Compute handwriting features per sample ID and save them to a CSV file.

    Input
    -----
    full_df : pandas.DataFrame
        Concatenated point-level handwriting data containing an 'id' column that
        identifies each sample/trajectory. For each group (one id), the DataFrame
        must include the columns required by `extract_features`, e.g.:
        ['x', 'y', 'timestamp', 'pen_status', 'pressure'] (and optionally others).

    output_csv : str
        Path to the output CSV file where one row per 'id' will be saved.
        Parent directories are created automatically if they do not exist.

    Output
    ------
    None
        Writes a CSV file at `output_csv` with one row per sample (id) containing
        all extracted feature columns plus the 'id' column.

    """

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    all_features = []
    
    for id, row_group in full_df.groupby("id"):
        features = extract_features(row_group)
        features["id"] = id
        all_features.append(features)
        
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")            