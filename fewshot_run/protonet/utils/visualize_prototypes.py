import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from protonet.data.emothaw_dataset import EMOTHAWDataset
from protonet.models.protonet import ProtoNet, compute_prototypes


parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--data_root", default=None)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--log_dir", default="results")
parser.add_argument("--samples", type=int, default=10,
                    help="Number of samples per class to compute prototypes")
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
    Normalize([0.5]*3, [0.5]*3)
])

dataset = EMOTHAWDataset(data_root, transform)

# collect sample embeddings per class
class_to_idx = dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

images_by_class = {cls: [] for cls in class_to_idx.values()}

# pick sample images
for img, label in dataset:
    if len(images_by_class[label]) < args.samples:
        images_by_class[label].append(img.unsqueeze(0))
    if all(len(v) >= args.samples for v in images_by_class.values()):
        break

support_images = torch.cat([torch.cat(v, dim=0) for v in images_by_class.values()], dim=0)
support_labels = torch.tensor([
    cls for cls, v in images_by_class.items() for _ in range(len(v))
])

###############################################
# Model
###############################################
model_path = os.path.join(args.log_dir, f"protonet_{args.task}", "best_model.pt")
model = ProtoNet()
model.load_state_dict(torch.load(model_path, map_location=args.device))
model.to(args.device)
model.eval()

###############################################
# Compute prototypes
###############################################
with torch.no_grad():
    emb = model(support_images.to(args.device))
    emb = emb.cpu()

prototypes = compute_prototypes(emb, support_labels, n_way=len(class_to_idx), k_shot=args.samples)

###############################################
# Visualize prototype heatmaps
###############################################
prototypes = prototypes.view(prototypes.size(0), 8, 8)  # because Conv4 produces 64*8*8

plt.figure(figsize=(12, 4))
for i, proto in enumerate(prototypes):
    ax = plt.subplot(1, prototypes.size(0), i+1)
    plt.imshow(proto.detach().numpy(), cmap="viridis")
    plt.title(idx_to_class[i])
    plt.axis("off")

plt.suptitle(f"Prototype Heatmaps for {args.task}")
plt.tight_layout()
plt.savefig(f"{args.task}_prototype_heatmaps.png")
plt.show()
