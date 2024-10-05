
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
        case "ViTB16":
            model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            model.heads.head = torch.nn.Linear(768, SKETCHY_CLASSES)
        case "RegNetY128GF":
            model = torchvision.models.regnet_y_128gf(weights=torchvision.models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1)
            model.fc = torch.nn.Linear(7392, SKETCHY_CLASSES)
        case _:
            raise ValueError(f"Unknown architecture {architecture}")
    return model
