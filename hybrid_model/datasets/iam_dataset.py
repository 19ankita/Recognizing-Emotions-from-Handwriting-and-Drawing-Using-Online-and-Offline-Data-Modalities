import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class IAMDataset(Dataset):
    def __init__(
        self,
        metadata_csv,
        image_root="data/raw/IAM_OnDB/images",
        image_size=224
    ):
        self.df = pd.read_csv(metadata_csv)
        self.df.columns = self.df.columns.str.strip()
        self.image_root = image_root

        self.image_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def _to_xy_pen(self, traj: np.ndarray) -> np.ndarray:
        """
        Convert loaded trajectory to [T,3] = (x,y,pen).
        Supports:
          - [T,3] already (x,y,pen)  -> return as is
          - [T,4] (x,y,t,stroke_id)  -> pen=0 at start of each new stroke
          - [T,3] (x,y,t)            -> fallback pen=1 everywhere (not ideal)
        """
        if traj.ndim != 2:
            raise ValueError(f"Trajectory must be 2D, got shape {traj.shape}")

        T, C = traj.shape

        if C == 3:
            # Could be (x,y,pen) OR (x,y,t). We assume it's (x,y,pen) if last col is in [0,1] and mostly binary.
            last = traj[:, 2]
            # heuristic: if values look like 0/1
            if np.all((last >= 0.0) & (last <= 1.0)) and (np.mean((last == 0.0) | (last == 1.0)) > 0.9):
                return traj.astype(np.float32)  # (x,y,pen)
            else:
                # fallback treat as (x,y,t)
                pen = np.ones((T, 1), dtype=np.float32)
                return np.concatenate([traj[:, :2].astype(np.float32), pen], axis=1)

        if C == 4:
            # (x,y,t,stroke_id) -> make pen signal
            sid = traj[:, 3].astype(int)
            pen = np.ones((T,), dtype=np.float32)
            # pen-up at stroke changes
            pen[1:] = (sid[1:] == sid[:-1]).astype(np.float32)  # 0 when new stroke begins
            out = np.stack([traj[:, 0], traj[:, 1], pen], axis=1).astype(np.float32)
            return out

        raise ValueError(f"Unsupported trajectory shape {traj.shape}. Expected [T,3] or [T,4].")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image"]
        traj_path = row["trajectory"]

        img = Image.open(img_path).convert("L")
        img = self.image_transform(img)

        traj = np.load(traj_path)
        traj = self._to_xy_pen(traj)          # [T,3]
        traj = torch.from_numpy(traj)         # float32

        return img, traj