import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.models.reverse_model import ReverseModel

# ---------------------------
# CONFIG
# ---------------------------
EMOTHAW_ROOT = "data/EMOTHAW"
OUT_ROOT = "data/EMOTHAW/pseudo_trajectories"
MODEL_PATH = "checkpoint/reverse_model.pth"

TASK = "cursive_writing"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 200

# ---------------------------
# IMAGE PREPROCESSING
# ---------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------
# LOAD MODEL
# ---------------------------
model = ReverseModel(seq_len=SEQ_LEN)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------------------
# GENERATE PSEUDO-TRAJECTORIES
# ---------------------------
img_dir = os.path.join(EMOTHAW_ROOT, TASK)
out_dir = os.path.join(OUT_ROOT, TASK)

os.makedirs(out_dir, exist_ok=True)

image_files = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

print(f"[{TASK}] {len(image_files)} images")

for fname in image_files:
    img_path = os.path.join(img_dir, fname)

    img = Image.open(img_path).convert("L")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        traj_pred = model(img).cpu().squeeze().numpy()

    out_path = os.path.join(
        out_dir,
        fname.rsplit(".", 1)[0] + ".npy"
    )

    np.save(out_path, traj_pred)

print("Pseudo-trajectories generated for cursive_writing")



