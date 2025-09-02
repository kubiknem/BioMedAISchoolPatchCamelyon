from collections.abc import Iterable

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from patch_camelyon.typing import Input


class DataModule(LightningDataModule):
    def __init__(
        self, batch_size: int, num_workers: int = 0, **datasets: DictConfig
    ) -> None:
        # TODO: Implement the initializer for the DataModule.
        pass

    def setup(self, stage: str) -> None:
        # TODO: Implement the setup method.
        # This method is called by Lightning to set up the dataset(s) for a specific stage.
        #   - Use a `match` statement or `if/elif` block to check the value of the `stage` argument.
        #   - Use `hydra.utils.instantiate()` to create the dataset objects from their configs.
        pass

    def train_dataloader(self) -> Iterable[Input]:
        # TODO: Create and return the DataLoader for the training set.
        #   - Correctly setup the workers! Either read the docs, or discuss with agents.
        pass

    def val_dataloader(self) -> Iterable[Input]:
        # TODO: Create and return the DataLoader for the validation set.
        pass

    def test_dataloader(self) -> Iterable[Input]:
        # TODO: Create and return the DataLoader for the test set.
        pass
