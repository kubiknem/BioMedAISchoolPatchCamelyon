from torch import Tensor, nn


class BinaryClassifier(nn.Module):
    def __init__(self) -> None:
        # TODO: Implement the initializer for the classification head.
        # This module will take the feature map from a backbone and produce a final prediction.
        #  - We suggest using global average pooling and droppout.
        pass

    def forward(self, x: Tensor) -> Tensor:
        # TODO: Implement the forward pass for the classifier.
        # The input `x` is the feature map from the backbone with shape (Batch, Channels, Height, Width).
        #   - Don't forget to apply relevant activation function to the output logits to get a probability.
        pass
