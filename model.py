import torch
from torch import nn
from torchvision import models

def create_model(num_classes: int):
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
