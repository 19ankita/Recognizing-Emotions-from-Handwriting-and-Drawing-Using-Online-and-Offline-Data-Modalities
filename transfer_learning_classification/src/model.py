import torch
import torch.nn as nn
from torchvision import models


# -------------------------------------------------------------------
# ResNet backbone extended to accept PSEUDO-DYNAMIC FEATURES
# (CLASSIFIER-ONLY TRAINING)
# -------------------------------------------------------------------
class ResNetWithDynamicFeatures(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True):
        super().__init__()

        # Load backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze ENTIRE backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classifier with Identity
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 5 pseudo-dynamic features → 32D embedding
        self.dynamic_head = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Final classifier (ONLY this + dynamic_head train)
        self.classifier = nn.Linear(in_features + 32, num_classes)

    def forward(self, x, pseudo_dyn):
        """
        x:          [B, 3, H, W]
        pseudo_dyn: [B, 5]
        """
        img_features = self.backbone(x)               # frozen
        dyn_features = self.dynamic_head(pseudo_dyn)  # trainable

        fused = torch.cat([img_features, dyn_features], dim=1)
        return self.classifier(fused)


def build_resnet18(num_classes, freeze_backbone=True):
    return ResNetWithDynamicFeatures(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )


