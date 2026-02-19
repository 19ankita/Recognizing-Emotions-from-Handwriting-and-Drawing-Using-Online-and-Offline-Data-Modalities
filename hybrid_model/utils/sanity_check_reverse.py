import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets.iam_dataset import IAMDataset
from models.build_reverse_model import ReverseModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "reverse_model.pth"

dataset = IAMDataset("data/processed/IAM_OnDB/metadata.csv")

model = ReverseModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Take one sample
img, traj_gt = dataset[0]

with torch.no_grad():
    traj_pred = model(img.unsqueeze(0).to(DEVICE))
    traj_pred = traj_pred.squeeze(0).cpu().numpy()

traj_gt = traj_gt.numpy()

# Plot
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Ground Truth")
plt.plot(traj_gt[:,0], -traj_gt[:,1])
plt.axis("equal")

plt.subplot(1,2,2)
plt.title("Predicted")
plt.plot(traj_pred[:,0], -traj_pred[:,1])
plt.axis("equal")

plt.show()
