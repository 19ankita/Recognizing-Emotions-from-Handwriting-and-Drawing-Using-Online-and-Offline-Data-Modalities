import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.build_reverse_model import ReverseModel


def generate_pseudo_trajectories(
    emothaw_root="data/raw/EMOTHAW",
    out_root="data/processed/EMOTHAW/pseudo_trajectories",
    model_path="reverse_model.pth",
    task="cursive_writing",
    seq_len=200,
    device=None
):
    """
    Generate pseudo-trajectories from EMOTHAW images using trained reverse model.
    """

    # ---------------------------
    # DEVICE
    # ---------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------
    # IMAGE TRANSFORM
    # ---------------------------
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ---------------------------
    # LOAD MODEL
    # ---------------------------
    model = ReverseModel(seq_len=seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ---------------------------
    # PATHS
    # ---------------------------
    img_dir = os.path.join(emothaw_root, task)
    out_dir = os.path.join(out_root, task)

    os.makedirs(out_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"[{task}] {len(image_files)} images")

    # ---------------------------
    # INFERENCE LOOP
    # ---------------------------
    for fname in image_files:
        img_path = os.path.join(img_dir, fname)

        img = Image.open(img_path).convert("L")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            traj_pred = model(img).cpu().squeeze().numpy()

        out_path = os.path.join(
            out_dir,
            fname.rsplit(".", 1)[0] + ".npy"
        )

        np.save(out_path, traj_pred)

    print(f"Pseudo-trajectories generated for {task}")
