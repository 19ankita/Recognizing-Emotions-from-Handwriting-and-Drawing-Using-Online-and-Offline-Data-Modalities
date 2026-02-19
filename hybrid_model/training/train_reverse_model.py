import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import IAMDataset
from model import build_reverse_model


# ---------------------------
# CONFIG
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
LAMBDA_SMOOTH = 0.1  # set 0.0 to disable


# ---------------------------
# SMOOTHNESS LOSS
# ---------------------------
def smoothness_loss(traj):
    # penalize large velocity changes
    vel = traj[:, 1:, :2] - traj[:, :-1, :2]
    acc = vel[:, 1:] - vel[:, :-1]
    return acc.pow(2).mean()


# ---------------------------
# TRAINING
# ---------------------------
def run_train():
    dataset = IAMDataset("data/processed/IAM_OnDB/metadata.csv")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = build_reverse_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, traj_gt in loader:
            imgs = imgs.to(DEVICE)
            traj_gt = traj_gt.to(DEVICE)

            traj_pred = model(imgs)

            # Reconstruction loss (x, y only)
            mse = F.mse_loss(traj_pred[:, :, :2], traj_gt[:, :, :2])

            smooth = smoothness_loss(traj_pred)
            loss = mse + LAMBDA_SMOOTH * smooth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "reverse_model.pth")
    print("Model saved")


