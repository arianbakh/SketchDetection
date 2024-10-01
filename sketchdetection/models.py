
from sketchdetection.datasets import SKETCHY_CLASSES
import torch
import torchvision


def get_model(architecture: str):
    match architecture:
        case "ResNet152":
            model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
            model.fc = torch.nn.Linear(2048, SKETCHY_CLASSES)
        case "EfficientNetV2L":
            model = torchvision.models.efficientnet_v2_l(weights=torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            model.classifier[1] = torch.nn.Linear(1280, SKETCHY_CLASSES)
        case _:
            raise ValueError(f"Unknown architecture {architecture}")
    return model
