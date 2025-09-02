from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall

from patch_camelyon.modeling import BinaryClassifier
from patch_camelyon.typing import Input, Outputs


class PatchCamelyonModel(LightningModule):
    def __init__(self, backbone: nn.Module) -> None:
        # TODO: Implement the initializer for the main model.
        #   - Store the `backbone` module as an instance attribute.
        #   - Instantiate your `BinaryClassifier`.
        #   - Instantiate the appropriate loss function.
        #   - Set up metrics for validation using `torchmetrics.MetricCollection`.
        #     - Include `AUROC`, `Accuracy`, `Precision`, and `Recall`
        #     - Add a prefix, e.g., `prefix="validation/"`, for clear logging.
        #   -  Clone the validation metrics for the test set.
        pass

    def forward(self, x: Input) -> Outputs:
        # TODO: Implement the forward pass of the model.
        # This defines how input data flows through the network.
        pass

    def training_step(self, batch: Input) -> Tensor:
        # TODO: Implement a single training step.
        # This method is called for each batch during training.
        #   - Perform a forward pass on the `inputs` to get model `outputs`.
        #   - Calculate the loss.
        #   - Log the training loss using `self.log()`. Name it "train/loss" and set `prog_bar=True`.
        #   - Return the calculated loss.
        pass

    def validation_step(self, batch: Input) -> None:
        # TODO: Implement a single validation step.
        # This is called for each batch during validation.
        #   - Perform a forward pass on the `inputs` to get model `outputs`.
        #   - Calculate the valiadation loss.
        #   - Log the validation loss as "validation/loss" using `self.log()`, ensuring it's logged on epoch end.
        #   - Update the validation metrics.
        #   - Log the metrics dictionary using `self.log_dict(..., on_epoch=True)`.
        pass

    def test_step(self, batch: Input) -> None:
        # TODO: Implement a single test step.
        # This is called for each batch during testing.
        #   - Perform a forward pass on the `inputs` to get model `outputs`.
        #   - Update the test metrics.
        #   - Log the test metrics dictionary using `self.log_dict(..., on_epoch=True)`.
        pass

    def configure_optimizers(self) -> Optimizer:
        # TODO: Configure the optimizer for the model.
        #   - Choose an optimizer, for example, `AdamW`.
        #   - Pass the model's parameters to the optimizer.
        pass
