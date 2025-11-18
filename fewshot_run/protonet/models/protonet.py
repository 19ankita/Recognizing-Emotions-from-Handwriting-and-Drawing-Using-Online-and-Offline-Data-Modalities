import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)     # downsample by 2
        )

    def forward(self, x):
        return self.layer(x)


class ProtoNet(nn.Module):
    """
    Improved Conv-4 few-shot backbone.
    Used in miniImageNet, Omniglot, CIFAR-FS few-shot benchmarks.
    """

    def __init__(self, x_dim=3, hid_dim=64, z_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(x_dim, hid_dim),        # 64
            ConvBlock(hid_dim, hid_dim),      # 64
            ConvBlock(hid_dim, hid_dim * 2),  # 128
            ConvBlock(hid_dim * 2, z_dim),    # 128
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return x


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
