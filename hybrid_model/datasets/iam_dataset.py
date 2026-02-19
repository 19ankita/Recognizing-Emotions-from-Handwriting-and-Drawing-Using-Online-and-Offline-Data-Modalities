import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class IAMDataset(Dataset):
    def __init__(self, 
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image"]
        traj_path = row["trajectory"]

        img = Image.open(img_path).convert("L")
        img = self.image_transform(img)

        traj = np.load(traj_path)
        traj = torch.tensor(traj, dtype=torch.float32)

        return img, traj

