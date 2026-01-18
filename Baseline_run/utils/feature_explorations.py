from src.svc_reader import read_all_svc_files
from src.feature_utils import extract_features

import pandas as pd
import matplotlib.pyplot as plt

data = read_all_svc_files("Baseline_run/dataset/words")

sample_id, arr = next(iter(data.items()))
df = pd.DataFrame(arr, columns=["x", "y", "timestamp", "pen_status", "azimuth", "altitude", "pressure"])

df["id"] = sample_id

features = extract_features(df)

for k,v in features.items():
    print(f"{k:30s} : {v:.4f}")
    

