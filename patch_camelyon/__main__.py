from random import randint

import hydra
from lightning import seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import Trainer, autolog

from patch_camelyon.data import DataModule
from patch_camelyon.patch_camelyon_model import PatchCamelyonModel


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


# TODO: Fill the right arguments to the hydra decorator
@hydra.main(...)
@autolog
def main(config: DictConfig, logger: Logger | None) -> None:
    # TODO: Implement the main training and evaluation script.
    # This function orchestrates the entire ML pipeline using the provided configuration.

    # For reproducibility, seed everything
    seed_everything(config.seed, workers=True)

    # TODO: **Instantiate the DataModule**: Create an instance of your `DataModule`.
    #  - Use `hydra.utils.instantiate()` with proper parameters.
    #  - Set `_recursive_=False` to prevent Hydra from instantiating the nested dataset
    #      configs immediately; the DataModule's `.setup()` method will handle that.
    data = ...

    # TODO: **Instantiate the Model**: Create an instance of your `PatchCamelyonModel`.
    #  - Use `hydra.utils.instantiate()` with the model configuration.
    model = ...

    # TODO: **Instantiate the Trainer**: Create the Lightning `Trainer`.
    #  - Use `hydra.utils.instantiate()` with the trainer configuration.
    #  - Pass the `logger` object provided to this main function.
    trainer = ...

    # Dynamically gets desired trainer method (mode) (e.g., 'fit' or 'test')
    getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
