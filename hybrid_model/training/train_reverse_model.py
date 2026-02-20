import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.iam_dataset import IAMDataset
from models.build_reverse_model import ReverseModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4

LAMBDA_SMOOTH = 0.1
LAMBDA_PEN = 1.0  # tune 0.5–2.0


def smoothness_loss_xy(traj_xy, pen_mask=None):
    """
    traj_xy: [B,T,2]
    pen_mask: [B,T,1] with 1 where pen_down else 0 (optional)
    """
    vel = traj_xy[:, 1:] - traj_xy[:, :-1]          # [B,T-1,2]
    acc = vel[:, 1:] - vel[:, :-1]                  # [B,T-2,2]

    if pen_mask is not None:
        # only enforce smoothness when pen is down in consecutive steps
        m = pen_mask[:, 2:] * pen_mask[:, 1:-1] * pen_mask[:, :-2]  # [B,T-2,1]
        return (acc.pow(2) * m).sum() / (m.sum() + 1e-8)

    return acc.pow(2).mean()


def run_train():
    dataset = IAMDataset("data/processed/IAM_OnDB/metadata.csv")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ReverseModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        total_xy = 0.0
        total_pen = 0.0

        for imgs, traj_gt in loader:
            imgs = imgs.to(DEVICE)
            traj_gt = traj_gt.to(DEVICE)   # expected [B,T,3] = x,y,pen(0/1)

            pred = model(imgs)             # must be [B,T,3] = x,y,pen_logit

            pred_xy = pred[:, :, :2]
            pred_pen_logit = pred[:, :, 2]         # [B,T]

            gt_xy = traj_gt[:, :, :2]
            gt_pen = traj_gt[:, :, 2]              # [B,T] in {0,1}
            pen_mask = gt_pen.unsqueeze(-1)        # [B,T,1]

            print(pred.shape, traj_gt.shape, gt_pen.min().item(), gt_pen.max().item())
            break
        
            # XY loss only where pen is down
            xy_loss = (F.mse_loss(pred_xy * pen_mask, gt_xy * pen_mask, reduction="sum")
                       / (pen_mask.sum() + 1e-8))

            # Pen classification loss
            pen_loss = F.binary_cross_entropy_with_logits(pred_pen_logit, gt_pen)

            # Smoothness on xy (masked)
            smooth = smoothness_loss_xy(pred_xy, pen_mask=pen_mask)

            loss = xy_loss + LAMBDA_PEN * pen_loss + LAMBDA_SMOOTH * smooth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()
            total_xy += xy_loss.item()
            total_pen += pen_loss.item()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {total/len(loader):.4f} | "
            f"XY: {total_xy/len(loader):.4f} | "
            f"PEN: {total_pen/len(loader):.4f}"
        )

    torch.save(model.state_dict(), "reverse_model.pth")
    print("Model saved")