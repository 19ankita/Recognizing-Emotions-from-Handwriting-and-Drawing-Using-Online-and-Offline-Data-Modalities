import os
import json
import csv
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from protonet.models.protonet import prototypical_loss


class ProtoEngine:
    def __init__(self, model, optimizer=None, lr=1e-3, device="cpu", log_dir="results"):

        self.model = model.to(device)
        self.device = device

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        # Logging folders
        os.makedirs(log_dir, exist_ok=True)
        self.trace_file = os.path.join(log_dir, "trace.txt")
        self.csv_file = os.path.join(log_dir, "metrics.csv")
        self.model_file = os.path.join(log_dir, "best_model.pt")

        # Clear previous logs
        if os.path.exists(self.trace_file):
            os.remove(self.trace_file)
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)

        # CSV header
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "acc", "split"])

        self.best_val_loss = float("inf")

        # TensorBoard
        self.writer = None

    def _split_episode(self, batch, n_way, k_shot, q_query):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        support_idx, query_idx = [], []

        for cls in range(n_way):
            start = cls * (k_shot + q_query)
            support_idx.extend(range(start, start + k_shot))
            query_idx.extend(range(start + k_shot, start + k_shot + q_query))

        support_idx = torch.tensor(support_idx, device=self.device)
        query_idx = torch.tensor(query_idx, device=self.device)

        support_images = images[support_idx]
        query_images = images[query_idx]

        support_labels = labels[support_idx]

        # Remap labels to 0..N-1
        uniq = support_labels.unique().tolist()
        remap = {old: new for new, old in enumerate(uniq)}

        support_labels = torch.tensor([remap[int(l)] for l in support_labels], device=self.device)
        query_labels = torch.tensor([remap[int(labels[i])] for i in query_idx], device=self.device)

        return support_images, support_labels, query_images, query_labels

    def train(self, train_loader, val_loader, n_way, k_shot, q_query, max_epochs):

        for epoch in range(1, max_epochs + 1):

            self.model.train()
            train_loss = 0
            train_acc = 0
            steps = 0

            for batch in train_loader:
                s_img, s_lbl, q_img, q_lbl = self._split_episode(batch, n_way, k_shot, q_query)

                self.optimizer.zero_grad()

                s_emb = self.model(s_img)
                q_emb = self.model(q_img)

                loss, acc = prototypical_loss(s_emb, s_lbl, q_emb, q_lbl, n_way)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_acc += acc.item()
                steps += 1

            train_loss /= steps
            train_acc /= steps

            # Validation
            self.model.eval()
            val_loss = 0
            val_acc = 0
            vsteps = 0

            with torch.no_grad():
                for batch in val_loader:
                    s_img, s_lbl, q_img, q_lbl = self._split_episode(batch, n_way, k_shot, q_query)
                    s_emb = self.model(s_img)
                    q_emb = self.model(q_img)

                    loss, acc = prototypical_loss(s_emb, s_lbl, q_emb, q_lbl, n_way)

                    val_loss += loss.item()
                    val_acc += acc.item()
                    vsteps += 1

            val_loss /= vsteps
            val_acc /= vsteps

            # Record trace
            trace = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }

            with open(self.trace_file, "a") as f:
                f.write(json.dumps(trace) + "\n")

            # CSV
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, train_acc, "train"])
                writer.writerow([epoch, val_loss, val_acc, "val"])

            # TensorBoard
            if self.writer:
                self.writer.add_scalar("loss/train", train_loss, epoch)
                self.writer.add_scalar("loss/val", val_loss, epoch)
                self.writer.add_scalar("accuracy/train", train_acc, epoch)
                self.writer.add_scalar("accuracy/val", val_acc, epoch)

            print(f"[Epoch {epoch}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_file)
                print(f"Saved best model (val_loss = {val_loss:.4f})")
