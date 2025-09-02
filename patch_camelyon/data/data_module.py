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
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets
        

    def setup(self, stage: str) -> None:
        # TODO: Implement the setup method.
        # This method is called by Lightning to set up the dataset(s) for a specific stage.
        #   - Use a `match` statement or `if/elif` block to check the value of the `stage` argument.
        #   - Use `hydra.utils.instantiate()` to create the dataset objects from their configs.
        match stage:
            case "fit":
                self.train = PatchCamelyon(
                    path_x=TRAIN_X_PATH,
                    path_y=TRAIN_Y_PATH,
                    transforms=train_transforms,
                )
                self.val = PatchCamelyon(
                    path_x=VALID_X_PATH,
                    path_y=VALID_Y_PATH,
                    transforms=test_transforms,
                )
            case "validate":
                self.val = PatchCamelyon(
                    path_x=VALID_X_PATH,
                    path_y=VALID_Y_PATH,
                    transforms=test_transforms,
                )
            case "test":
                self.test = PatchCamelyon(
                    path_x=TEST_X_PATH,
                    path_y=TEST_Y_PATH,
                    transforms=test_transforms,
                )
        

    def train_dataloader(self) -> Iterable[Input]:
        # TODO: Create and return the DataLoader for the training set.
        #   - Correctly setup the workers! Either read the docs, or discuss with agents.
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Input]:
        # TODO: Create and return the DataLoader for the validation set.
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[Input]:
        # TODO: Create and return the DataLoader for the test set.
                return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )
