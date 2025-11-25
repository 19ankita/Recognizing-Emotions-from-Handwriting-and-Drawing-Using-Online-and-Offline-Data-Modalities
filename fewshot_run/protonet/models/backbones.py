import torch
import torch.nn as nn
import torchvision.models as models


# -------------------------------------------------------
# Conv6 backbone (for few-shot learning)
# -------------------------------------------------------
class Conv6Backbone(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.encoder = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 256),
            conv_block(256, 128),
            conv_block(128, 128),
        )

        self.proj = nn.Linear(128 * 1 * 1, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


# -------------------------------------------------------
# Conv8 backbone
# -------------------------------------------------------
class Conv8Backbone(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        channels = [3, 64, 64, 128, 128, 256, 256, 128, 128]

        blocks = []
        for i in range(8):
            blocks.append(conv_block(channels[i], channels[i+1]))

        self.encoder = nn.Sequential(*blocks)
        self.proj = nn.Linear(128 * 1 * 1, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


# -------------------------------------------------------
# ResNet18 Encoder (for few-shot learning)
# -------------------------------------------------------
class ResNet18Encoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()

        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()   # remove classifier
        self.backbone = backbone

        self.proj = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        return self.proj(x)
