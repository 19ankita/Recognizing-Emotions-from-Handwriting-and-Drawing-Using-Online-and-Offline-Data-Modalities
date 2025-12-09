import torch.nn as nn
import torch
from torchvision import models

def freeze_resnet_layers(model):
    """
    Freeze conv1, bn1, layer1, layer2, layer3
    Keep layer4 + fc trainable
    """
    # Freeze initial layers
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False

    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = False

    # KEEP layer4 trainable
    for p in model.layer4.parameters():
        p.requires_grad = True

    # Classifier always trainable
    for p in model.fc.parameters():
        p.requires_grad = True

    return model


def build_resnet18(num_classes, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    # Freeze only layers 1–3
    if freeze_backbone:
        model = freeze_resnet_layers(model)
        
    return model


def build_resnet50(num_classes, freeze_backbone=True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze only layers 1–3
    if freeze_backbone:
        model = freeze_resnet_layers(model)
    return model