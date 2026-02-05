import torch
import torch.nn as nn
from torchvision import models


# -------------------------------------------------------------------
# ResNet backbone extended to accept PSEUDO-DYNAMIC FEATURES
# with bounded regression head (sigmoid)
# -------------------------------------------------------------------
class ResNetWithDynamicFeatures(nn.Module):
    def __init__(self, output_dim=4, freeze_backbone=True):
        super().__init__()

        # Load the pre-trained model
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze ENTIRE backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classifier with Identity
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # pseudo-dynamic features → 32D embedding
        self.dynamic_head = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Regression head
        self.regressor = nn.Linear(in_features + 32, output_dim)
        
        # Sigmoid for bounded outputs
        self.activation = nn.Sigmoid()

    def forward(self, x, pseudo_dyn):
        
        """
        Parameters
        ----------
        x : torch.Tensor
            Handwriting images of shape [B, 3, H, W]
        pseudo_dyn : torch.Tensor
            Pseudo-dynamic features of shape [B, 5]

        Returns
        -------
        torch.Tensor
            Predicted emotion scores of shape [B, output_dim]
        """
        img_features = self.backbone(x)               # frozen
        dyn_features = self.dynamic_head(pseudo_dyn)  # trainable

        fused = torch.cat([img_features, dyn_features], dim=1)
        
        # Regression + bounding
        out = self.regressor(fused)
        out = self.activation(out)
        
        return out


def build_resnet18(output_dim=4, freeze_backbone=True):
    
    """
    Build ResNet-18 based regression model with pseudo-dynamic features.
    """
    return ResNetWithDynamicFeatures(
        output_dim=output_dim,
        freeze_backbone=freeze_backbone
    )


