import torchvision
from torch import nn


def vgg16(weights: str | None = None) -> nn.Module:
    # TODO: Create a VGG-16 feature extractor.
    # We want to use a pretrained VGG-16 model, but only its convolutional layers.
    #   - Load the `vgg16` model from `torchvision.models`.
    #   - Extract only the convolutional layers.
    pass
