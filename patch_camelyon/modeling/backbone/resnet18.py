import torchvision
from torch import nn


def resnet18(weights: str | None = None) -> nn.Module:
    # TODO: Create a ResNet-18 feature extractor.
    # The goal is to use a pretrained ResNet-18 model but without its final classification layer.
    #   - Load the `resnet18` model from `torchvision.models`.
    #   - Remove the last two layers (the adaptive average pooling and the fully connected classifier).
    pass
