import torch
import torch.nn as nn
from torchvision import models

def freeze_low_level_layers(model):
    # Freeze ONLY conv1 + bn1
    for name, param in model.named_parameters():
        if name.startswith("conv1") or name.startswith("bn1"):
            param.requires_grad = False


# -------------------------------------------------------------------
# ResNet backbone extended to accept PSEUDO-DYNAMIC FEATURES
# -------------------------------------------------------------------
class ResNetWithDynamicFeatures(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=5, freeze_backbone=True):
        super().__init__()

        # Load backbone
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze only conv1 + bn1 (your requirement)
        if freeze_backbone:
            freeze_low_level_layers(base)

        # Replace classifier with Identity so we get pure features
        in_features = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base

        # 5 pseudo-dynamic features → 32D embedding
        self.dynamic_head = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Final classifier takes: CNN_features + dynamic_features
        self.classifier = nn.Linear(in_features + 32, num_classes)

    def forward(self, x, pseudo_dyn):
        """
        x:       image tensor  [B, 3, H, W]
        pseudo_dyn: float tensor [B, 5]
        """
        img_features = self.backbone(x)           # [B, 512] or [B, 2048]
        dyn_features = self.dynamic_head(pseudo_dyn)  # [B, 32]

        fused = torch.cat([img_features, dyn_features], dim=1)

        return self.classifier(fused)


# -------------------------------------------------------------------
# Builder functions
# -------------------------------------------------------------------
def build_resnet18(num_classes, freeze_backbone=True):
    return ResNetWithDynamicFeatures(
        backbone="resnet18",
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )


def build_resnet50(num_classes, freeze_backbone=True):
    return ResNetWithDynamicFeatures(
        backbone="resnet50",
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )