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
        z = self.fc(x)
        return z


class TrajectoryDecoder(nn.Module):
    def __init__(self, latent_dim=128, seq_len=200):
        super().__init__()

        self.seq_len = seq_len

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len * 3)
        )

    def forward(self, z):
        out = self.decoder(z)
        return out.view(-1, self.seq_len, 3)


class ReverseModel(nn.Module):
    def __init__(self, latent_dim=128, seq_len=200):
        super().__init__()
        self.encoder = CNNEncoder(latent_dim)
        self.decoder = TrajectoryDecoder(latent_dim, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        traj = self.decoder(z)
        return traj
