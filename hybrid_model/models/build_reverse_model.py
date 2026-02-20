import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)


class TrajectoryDecoder(nn.Module):
    def __init__(self, latent_dim=128, seq_len=200):
        super().__init__()
        self.seq_len = seq_len

        self.shared = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
        )

        # head for x,y
        self.xy_head = nn.Linear(512, seq_len * 2)

        # head for pen (logits)
        self.pen_head = nn.Linear(512, seq_len * 1)

    def forward(self, z):
        h = self.shared(z)
        xy = self.xy_head(h).view(-1, self.seq_len, 2)
        pen_logit = self.pen_head(h).view(-1, self.seq_len, 1)
        return torch.cat([xy, pen_logit], dim=-1)  # [B,T,3]


class ReverseModel(nn.Module):
    def __init__(self, latent_dim=128, seq_len=200):
        super().__init__()
        self.encoder = CNNEncoder(latent_dim)
        self.decoder = TrajectoryDecoder(latent_dim, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)