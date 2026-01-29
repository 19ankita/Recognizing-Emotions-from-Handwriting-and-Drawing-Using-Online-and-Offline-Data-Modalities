import torch
import torch.nn as nn
from torchvision import models


# -------------------------------------------------------------------
# ResNet backbone extended to accept PSEUDO-DYNAMIC FEATURES
# (CLASSIFIER-ONLY TRAINING)
# -------------------------------------------------------------------
class ResNetWithDynamicFeatures(nn.Module):
    def __init__(self, backbone="resnet18", output_dim=4, freeze_backbone=True):
        super().__init__()

        # Load backbone
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze ENTIRE backbone
        if freeze_backbone:
            for param in base.parameters():
                param.requires_grad = False

        # Replace classifier with Identity
        in_features = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base

        # pseudo-dynamic features → 32D embedding
        self.dynamic_head = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Regression head
        self.regressor = nn.Linear(in_features + 32, output_dim)

    def forward(self, x, pseudo_dyn):
        """
        x:          [B, 3, H, W]
        pseudo_dyn: [B, 5]
        """
        img_features = self.backbone(x)               # frozen
        dyn_features = self.dynamic_head(pseudo_dyn)  # trainable

        fused = torch.cat([img_features, dyn_features], dim=1)
        return self.regressor(fused)


def build_resnet18(output_dim=4, freeze_backbone=True):
    return ResNetWithDynamicFeatures(
        backbone="resnet18",
        output_dim=output_dim,
        freeze_backbone=freeze_backbone
    )


