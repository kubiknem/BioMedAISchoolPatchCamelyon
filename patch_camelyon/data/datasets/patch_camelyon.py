import numpy as np
import torch
from albumentations import TransformType
from albumentations.pytorch import ToTensorV2
from numpy.typing import NDArray
from torch.utils.data import Dataset

from patch_camelyon.typing import Sample


class PatchCamelyon(Dataset[Sample]):
    def __init__(
        self, path_x: str, path_y: str, transforms: TransformType | None = None
    ) -> None:
        # TODO: Implement the initializer for the dataset.
        #   - Lazy-load the image data from 'path_x' and label data from 'path_y'.
        super().__init__()
        self.x = np.load(path_x, mmap_mode="r")
        self.y = np.load(path_y, mmap_mode="r")
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
            # TODO: Implement the __len__ method.
            # This method should return the total number of samples in the dataset.
            return len(self.x)

    def __getitem__(self, index: int) -> Sample:
        # TODO: Implement the __getitem__ method.
        # This method should retrieve a single sample (image and label) from the dataset at the given index.
        #   - Check if any transforms are provided. If so, apply them to the image.
        #   - Convert both the image and the label to PyTorch tensors.
        image: NDArray[np.uint8] = self.x[index]  # (96, 96, 3)
        label: NDArray[np.uint8] = self.y[index]  # (1,)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        return self.to_tensor(image=image)["image"], torch.tensor(label).float()
