from torch import Tensor, nn


class BinaryClassifier(nn.Module):
    def __init__(self) -> None:
        # TODO: Implement the initializer for the classification head.
        # This module will take the feature map from a backbone and produce a final prediction.
        #  - We suggest using global average pooling and droppout.
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.proj = nn.Linear(512, 1)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: Implement the forward pass for the classifier.
        # The input `x` is the feature map from the backbone with shape (Batch, Channels, Height, Width).
        #   - Don't forget to apply relevant activation function to the output logits to get a probability.
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = x.flatten(start_dim=-3, end_dim=-1)  # (B, C)
        x = self.dropout(x)
        x = self.proj(x)
        return x.sigmoid()
