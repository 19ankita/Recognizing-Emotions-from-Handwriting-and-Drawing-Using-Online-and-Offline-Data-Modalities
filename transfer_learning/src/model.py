import torch.nn as nn
import torch
from torchvision import models

def build_resnet18(num_classes, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet50(num_classes, freeze_backbone=True):
    # Load pretrained ResNet-50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze backbone if required
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model