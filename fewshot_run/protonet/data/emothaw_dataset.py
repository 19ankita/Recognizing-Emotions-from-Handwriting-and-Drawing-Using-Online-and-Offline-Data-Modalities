import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class EMOTHAWDataset(Dataset):
    """
    Loads all samples for a single EMOTHAW task (pentagon, house, tree, spiral, words).
    Expects structure:
        root/
            depression/
            anxiety/
            stress/
    """

    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        self.samples = []
        self.class_to_idx = {}

        class_names = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        if len(class_names) == 0:
            raise ValueError(f"No class folders found in {root}")

        # Map depression/anxiety/stress → 0/1/2
        self.class_to_idx = {
            cls: i for i, cls in enumerate(class_names)
        }

        # Load all file paths
        for cls_name in class_names:
            cls_idx = self.class_to_idx[cls_name]
            cls_dir = os.path.join(root, cls_name)
            images = sorted(glob.glob(os.path.join(cls_dir, "*.png")))
            for img in images:
                self.samples.append((img, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = ToTensor()(img)

        return img, label
