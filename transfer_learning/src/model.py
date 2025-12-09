import torch.nn as nn
import torch
from torchvision import models

def build_resnet18(num_classes, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    if freeze_backbone:
        model.requires_grad_(False)        # freeze ALL layers
        model.fc.requires_grad_(True)      # keep classifier trainable
        
    return model


def build_resnet50(num_classes, freeze_backbone=True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        model.requires_grad_(False)
        model.fc.requires_grad_(True)

    return model