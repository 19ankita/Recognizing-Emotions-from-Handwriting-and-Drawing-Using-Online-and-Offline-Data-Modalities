import os
import json
import csv
import time
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from protonet.models.protonet import prototypical_loss


class ProtoEngine:

    def __init__(self, model, optimizer=None, lr=1e-3, device="cpu"):
        self.model = model.to(device)
        self.device = device

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        self.best_val_loss = float("inf")
        self.writer = None

    # -------------------------------------------------------
    # Utility: Create results folder for each task
    # -------------------------------------------------------
    def _init_experiment(self, task_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join("results", task_name, timestamp)

        os.makedirs(base, exist_ok=True)

        self.exp_dir = base
        self.trace_file = os.path.join(base, "trace.jsonl")
        self.csv_file = os.path.join(base, "metrics.csv")
        self.model_file = os.path.join(base, "best_model.pt")
        self.plot_file = os.path.join(base, "plots.png")
        self.gpu_log = os.path.join(base, "gpu_mem.txt")

        # TensorBoard folder
        self.writer = SummaryWriter(os.path.join(base, "tensorboard"))

        # initialize CSV
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "acc", "split"])

        return base

    # -------------------------------------------------------
    # Utility: save JSON trace per epoch
    # -------------------------------------------------------
    def _log_trace(self, data_dict):
        with open(self.trace_file, "a") as f:
            f.write(json.dumps(data_dict) + "\n")

    # -------------------------------------------------------
    # Utility: GPU memory logging
    # -------------------------------------------------------
    def _log_gpu_memory(self, epoch):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)

            with open(self.gpu_log, "a") as f:
                f.write(f"Epoch {epoch}: allocated={allocated:.2f}MB, reserved={reserved:.2f}MB\n")

    # -------------------------------------------------------
    # Utility: Plot curves at the end
    # -------------------------------------------------------
    def _plot_curves(self, history):
        epochs = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.plot(epochs, history["val_loss"], label="Val Loss")
        plt.legend(); plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history["train_acc"], label="Train Acc")
        plt.plot(epochs, history["val_acc"], label="Val Acc")
        plt.legend(); plt.title("Accuracy")

        plt.savefig(self.plot_file)
        plt.close()

    # -------------------------------------------------------
    # Split one episode
    # -------------------------------------------------------
    def _split_episode(self, batch, n_way, k_shot, q_query):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        # ---- group images by class (does NOT assume sorted dataset) ----
        unique_classes = torch.unique(labels)

        if len(unique_classes) != n_way:
            print(f"[ERROR] Sampler produced {len(unique_classes)} classes but expected {n_way}")
            raise ValueError("Sampler class mismatch")

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for new_cls_id, original_class in enumerate(unique_classes):

            cls_mask = (labels == original_class).nonzero().squeeze()

            # Check minimum samples
            required = k_shot + q_query
            if cls_mask.numel() < required:
                print(f"[ERROR] Class {original_class.item()} has only {cls_mask.numel()} samples; "
                    f"requires {required}")
                raise ValueError("Not enough samples per class in episode.")

            # Select support/query indices
            cls_indices = cls_mask[:required]
            s_idx = cls_indices[:k_shot]
            q_idx = cls_indices[k_shot:required]

            # Store images
            support_images.append(images[s_idx])
            query_images.append(images[q_idx])

            # Labels must be remapped to 0..n_way-1
            support_labels.append(torch.full((k_shot,), new_cls_id, device=self.device))
            query_labels.append(torch.full((q_query,), new_cls_id, device=self.device))

        # Concatenate all classes
        support_images = torch.cat(support_images, dim=0)
        support_labels = torch.cat(support_labels, dim=0)
        query_images = torch.cat(query_images, dim=0)
        query_labels = torch.cat(query_labels, dim=0)

        return support_images, support_labels, query_images, query_labels


    # -------------------------------------------------------
    # TRAIN TASK
    # -------------------------------------------------------
    def train_task(self, task_name, train_loader, val_loader,
                   n_way=3, k_shot=5, q_query=10,
                   max_epochs=60):

        self._init_experiment(task_name)
        history = {"train_loss": [], "train_acc": [],
                   "val_loss": [], "val_acc": []}

        print(f"\n========== TRAINING TASK: {task_name.upper()} ==========\n")
        start_time = time.time()

        for epoch in range(1, max_epochs + 1):

            epoch_start = time.time()

            # ------------------------
            # TRAINING
            # ------------------------
            self.model.train()
            train_loss = 0
            train_acc = 0
            steps = 0

            train_bar = tqdm(train_loader, desc=f"[{task_name}] Epoch {epoch}/{max_epochs} TRAIN",
                             leave=False)

            for batch in train_bar:
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

                train_bar.set_postfix(loss=loss.item(), acc=acc.item())

            train_loss /= steps
            train_acc /= steps

            # ------------------------
            # VALIDATION
            # ------------------------
            self.model.eval()
            val_loss = 0
            val_acc = 0
            vsteps = 0

            val_bar = tqdm(val_loader, desc=f"[{task_name}] Epoch {epoch}/{max_epochs} VAL",
                           leave=False)

            with torch.no_grad():
                for batch in val_bar:
                    s_img, s_lbl, q_img, q_lbl = self._split_episode(batch, n_way, k_shot, q_query)

                    s_emb = self.model(s_img)
                    q_emb = self.model(q_img)

                    loss, acc = prototypical_loss(s_emb, s_lbl, q_emb, q_lbl, n_way)

                    val_loss += loss.item()
                    val_acc += acc.item()
                    vsteps += 1

                    val_bar.set_postfix(loss=loss.item(), acc=acc.item())

            val_loss /= vsteps
            val_acc /= vsteps

            # ------------------------
            # LOGGING
            # ------------------------
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # JSON trace
            self._log_trace({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            # CSV
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, train_acc, "train"])
                writer.writerow([epoch, val_loss, val_acc, "val"])

            # TensorBoard
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("acc/train", train_acc, epoch)
            self.writer.add_scalar("acc/val", val_acc, epoch)

            # GPU memory log
            self._log_gpu_memory(epoch)

            # ------------------------
            # ETA CALCULATION
            # ------------------------
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            remaining = (max_epochs - epoch) * epoch_time
            finish_time = datetime.now() + timedelta(seconds=remaining)

            print(f"\n----- {task_name.upper()} Epoch {epoch}/{max_epochs} -----")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"Epoch Time: {epoch_time/60:.2f} min")
            print(f"Elapsed:    {elapsed/60:.2f} min")
            print(f"ETA:        {remaining/60:.2f} min (~{finish_time.strftime('%Y-%m-%d %H:%M:%S')})")

            # ------------------------
            # SAVE BEST MODEL
            # ------------------------
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_file)
                print(f"Saved best model (val_loss={val_loss:.4f})")

        # After all epochs
        print(f"\n>>> FINISHED TASK: {task_name.upper()}")
        self._plot_curves(history)

        return history, self.exp_dir

    # -------------------------------------------------------
    # TEST EVALUATION (few-shot episodes)
    # -------------------------------------------------------
    def evaluate(self, test_loader, n_way, k_shot, q_query):

        self.model.eval()
        total_loss = 0
        total_acc = 0
        steps = 0

        with torch.no_grad():
            for batch in test_loader:
                s_img, s_lbl, q_img, q_lbl = self._split_episode(
                    batch, n_way, k_shot, q_query
                )

                s_emb = self.model(s_img)
                q_emb = self.model(q_img)

                loss, acc = prototypical_loss(s_emb, s_lbl, q_emb, q_lbl, n_way)

                total_loss += loss.item()
                total_acc += acc.item()
                steps += 1

        avg_loss = total_loss / steps
        avg_acc = total_acc / steps

        # Save test accuracy into trace.jsonl
        self._log_trace({
            "test_loss": avg_loss,
            "test_acc": avg_acc
        })

        print(f"\n[Test Evaluation] Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

        return avg_loss, avg_acc
