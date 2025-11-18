import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from protonet.data.emothaw_dataset import EMOTHAWDataset
from protonet.models.protonet import ProtoNet


parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--data_root", default=None)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--log_dir", default="results")
args = parser.parse_args()


###############################################
# Dataset
###############################################
if args.data_root:
    data_root = args.data_root
else:
    data_root = f"emothaw_tasks/{args.task}"

transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize([0.5, 0.5, 0.5],
              [0.5, 0.5, 0.5])
])

dataset = EMOTHAWDataset(data_root, transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)


###############################################
# Load model
###############################################
model_path = os.path.join(args.log_dir, f"protonet_{args.task}", "best_model.pt")
model = ProtoNet()
model.load_state_dict(torch.load(model_path, map_location=args.device))
model.to(args.device)
model.eval()


###############################################
# Extract embeddings
###############################################
embeddings = []
labels = []

with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(args.device)
        emb = model(imgs)
        embeddings.append(emb.cpu())
        labels.append(lbls)

embeddings = torch.cat(embeddings, dim=0).numpy()
labels = torch.cat(labels, dim=0).numpy()

print("Embeddings shape:", embeddings.shape)


###############################################
# t-SNE
###############################################
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
emb2d = tsne.fit_transform(embeddings)

plt.figure(figsize=8, 8)
scatter = plt.scatter(emb2d[:, 0], emb2d[:, 1], c=labels, cmap="Set1")
plt.legend(*scatter.legend_elements(), title="Class")
plt.title(f"t-SNE Embeddings for {args.task}")
plt.savefig(f"{args.task}_tsne.png")
plt.show()
