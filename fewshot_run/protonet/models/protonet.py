import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """A basic convolutional block: Conv → BN → ReLU → MaxPool"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class ProtoNet(nn.Module):
    """
    Prototypical Network with Conv4 backbone:
    - Input: 3×128×128 RGB images (EMOTHAW)
    - Output: embedding vectors (64-dim by default)
    """

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(x_dim, hid_dim),       # 64 × 64 × 64
            ConvBlock(hid_dim, hid_dim),     # 64 × 32 × 32
            ConvBlock(hid_dim, hid_dim),     # 64 × 16 × 16
            ConvBlock(hid_dim, z_dim),       # 64 × 8 × 8
        )

    def forward(self, x):
        """
        Returns embedding vector for each sample.
        Flattening is performed only at the end.
        """
        x = self.encoder(x)
        return x.view(x.size(0), -1)


def compute_prototypes(embeddings, labels, n_way, k_shot):
    """
    Compute class prototypes as the mean embedding per class.

    embeddings: Tensor [N_support, embedding_dim]
    labels:     Tensor [N_support]
    """
    prototypes = []
    for cls in range(n_way):
        cls_embeddings = embeddings[labels == cls]
        proto = cls_embeddings.mean(dim=0)
        prototypes.append(proto)
    return torch.stack(prototypes)


def prototypical_loss(support_embeddings, support_labels,
                      query_embeddings, query_labels,
                      n_way):
    """
    Compute classification loss for an episodic batch.
    """

    # Compute prototypes
    prototypes = compute_prototypes(
        support_embeddings,
        support_labels,
        n_way=n_way,
        k_shot=support_labels.shape[0] // n_way
    )

    # Compute distances: (num_query × n_way)
    distances = torch.cdist(query_embeddings, prototypes)

    # Compute prediction using softmax over negative distances
    log_p_y = F.log_softmax(-distances, dim=1)

    loss = F.nll_loss(log_p_y, query_labels)
    _, y_hat = log_p_y.max(dim=1)
    acc = (y_hat == query_labels).float().mean()

    return loss, acc
