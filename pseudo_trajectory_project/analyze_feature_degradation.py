import torch
import pandas as pd
from src.datasets.iam_dataset import IAMDataset
from src.models.reverse_model import ReverseModel
from src.features.trajectory_features import extract_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = IAMDataset("data/IAM_OnDB/processed/metadata.csv")

model = ReverseModel().to(DEVICE)
model.load_state_dict(torch.load("reverse_model.pth"))
model.eval()

rows = []

for i in range(len(dataset)):
    img, traj_real = dataset[i]

    with torch.no_grad():
        traj_pred = model(img.unsqueeze(0).to(DEVICE)).cpu().squeeze().numpy()

    f_real = extract_features(traj_real.numpy())
    f_pred = extract_features(traj_pred)

    for k in f_real:
        rows.append({
            "feature": k,
            "real": f_real[k],
            "reconstructed": f_pred[k],
            "ratio": f_pred[k] / (f_real[k] + 1e-8)
        })

df = pd.DataFrame(rows)
df.to_csv("feature_degradation.csv", index=False)

print(df.groupby("feature")[["ratio"]].mean())
