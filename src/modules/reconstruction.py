import gin
import numpy as np
import torch
import pytorch_lightning as pl
import pl_bolts
import MinkowskiEngine as ME

from src.utils.metric import cal_psnr_from_mse


@gin.configurable
class LitReconstructionModuleBase(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        lr,
        momentum,
        weight_decay,
        warmup_steps_ratio,
        max_steps,
        best_metric_type,
        ignore_label=255,
        lr_eta_min=0.,
    ):
        super(LitReconstructionModuleBase, self).__init__()
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        self.criterion = torch.nn.MSELoss()
        self.best_metric_value = -np.inf if best_metric_type == "maximize" else np.inf

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=int(self.warmup_steps_ratio * self.max_steps),
            max_epochs=self.max_steps,
            eta_min=self.lr_eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def training_step(self, batch, batch_idx):
        in_data = self.prepare_input_data(batch)
        preds = self.model(in_data)
        loss = self.criterion(preds, batch["labels"])
        psnr = cal_psnr_from_mse(loss)
        self.log("train_loss", loss.item(), batch_size=batch["batch_size"], logger=True, prog_bar=True)
        self.log("train_psnr", psnr, batch_size=batch["batch_size"], logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        in_data = self.prepare_input_data(batch)
        preds = self.model(in_data)
        loss = self.criterion(preds, batch["labels"])
        psnr = cal_psnr_from_mse(loss)
        self.log("val_loss", loss.item(), batch_size=batch["batch_size"], logger=True, prog_bar=True)
        self.log("val_psnr", psnr, batch_size=batch["batch_size"], logger=True, prog_bar=True)
        return loss

    def prepare_input_data(self, batch):
        raise NotImplementedError


@gin.configurable
class LitReconMinkowskiModule(LitReconstructionModuleBase):
    def prepare_input_data(self, batch):
        in_data = ME.TensorField(
            features=batch["features"],
            coordinates=batch["coordinates"],
            quantization_mode=self.model.QMODE
        )
        return in_data