import torch.nn as nn
from torchvision import models

def freeze_low_level_layers(model):
    # Freeze ONLY conv1 + bn1
    for name, param in model.named_parameters():
        if name.startswith("conv1") or name.startswith("bn1"):
            param.requires_grad = False


def build_resnet18(num_classes, freeze_backbone=True):

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        freeze_low_level_layers(model)   # <-- Only conv1 + bn1 are frozen

    return model

def build_resnet50(num_classes, freeze_backbone=True):

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        freeze_low_level_layers(model)  # freeze conv1 + bn1 only

    return model
