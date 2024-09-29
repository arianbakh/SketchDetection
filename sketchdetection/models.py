
from sketchdetection.datasets import SKETCHY_CLASSES
import torch
import torchvision


def get_model():
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(2048, SKETCHY_CLASSES)
    return model
