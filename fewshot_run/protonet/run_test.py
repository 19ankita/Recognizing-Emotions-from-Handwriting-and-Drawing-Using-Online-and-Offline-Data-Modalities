import numpy as np
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from protonet.data.emothaw_dataset import EMOTHAWDataset
from protonet.data.episodic_sampler import EpisodicSampler
from protonet.models.protonet import ProtoNet, prototypical_loss


parser = argparse.ArgumentParser()

parser.add_argument("--task", type=str, required=True,
                    help="Task to test: pentagon, house, tree, spiral, words")

parser.add_argument("--data_root", type=str, default=None,
                    help="Override dataset root")

parser.add_argument("--way", type=int, default=3)
parser.add_argument("--shot", type=int, default=5)
parser.add_argument("--query", type=int, default=10)
parser.add_argument("--episodes", type=int, default=200,
                    help="Number of test episodes")

parser.add_argument("--log_dir", type=str, default="results")

parser.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()


###############################################
# Dataset location
###############################################
if args.data_root is not None:
    data_root = args.data_root
else:
    data_root = f"emothaw_tasks/{args.task}"

if not os.path.isdir(data_root):
    raise FileNotFoundError(data_root)


###############################################
# Load dataset
###############################################
transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize([0.5, 0.5, 0.5],
              [0.5, 0.5, 0.5])
])

dataset = EMOTHAWDataset(data_root, transform=transform)
labels = [lbl for _, lbl in dataset.samples]

sampler = EpisodicSampler(
    labels,
    n_way=args.way,
    k_shot=args.shot,
    q_query=args.query,
    episodes_per_epoch=args.episodes
)

loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)


###############################################
# Load trained model
###############################################
model_path = os.path.join(args.log_dir, f"protonet_{args.task}", "best_model.pt")

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = ProtoNet()
model.load_state_dict(torch.load(model_path, map_location=args.device))
model.to(args.device)
model.eval()

print(f"\nLoaded model from: {model_path}\n")


###############################################
# Episode helper
###############################################
def split_episode(batch):
    images, labels = batch
    images = images.to(args.device)
    labels = labels.to(args.device)

    support_idx = []
    query_idx = []
    for cls in range(args.way):
        start = cls * (args.shot + args.query)
        support_idx.extend(range(start, start + args.shot))
        query_idx.extend(range(start + args.shot, start + args.shot + args.query))

    support_idx = torch.tensor(support_idx, device=args.device)
    query_idx = torch.tensor(query_idx, device=args.device)

    support_imgs = images[support_idx]
    query_imgs = images[query_idx]

    # Remap labels
    support_labels = labels[support_idx]
    unique = support_labels.unique().tolist()
    remap = {old: new for new, old in enumerate(unique)}
    support_labels = torch.tensor([remap[int(l)] for l in support_labels], device=args.device)

    query_labels = torch.tensor([remap[int(labels[i])] for i in query_idx], device=args.device)

    return support_imgs, support_labels, query_imgs, query_labels


###############################################
# Evaluate
###############################################
all_loss = []
all_acc = []

with torch.no_grad():
    for batch in loader:
        s_img, s_lbl, q_img, q_lbl = split_episode(batch)

        s_emb = model(s_img)
        q_emb = model(q_img)

        loss, acc = prototypical_loss(s_emb, s_lbl, q_emb, q_lbl, args.way)
        all_loss.append(loss.item())
        all_acc.append(acc.item())

avg_loss = sum(all_loss) / len(all_loss)
avg_acc = sum(all_acc) / len(all_acc)

std_acc = np.std(all_acc)

print(f"=== TEST RESULTS for {args.task} ===")
print(f"Avg Loss: {avg_loss:.4f}")
print(f"Avg Acc:  {avg_acc*100:.2f}%")
print(f"Acc STD:  {std_acc*100:.2f}%\n")
