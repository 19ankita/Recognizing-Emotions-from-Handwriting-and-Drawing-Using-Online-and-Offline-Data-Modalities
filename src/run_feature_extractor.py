
import os
import pandas as pd
from .feature_utils import extract_features

def run_feature_extraction(full_df, output_csv):
    """Takes a concatenated DataFrame (with 'id' column) and computes features."""
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    all_features = []
    
    for id, row_group in full_df.groupby("id"):
        features = extract_features(row_group)
        features["id"] = id
        all_features.append(features)
        
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")            