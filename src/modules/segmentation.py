import gin
import numpy as np
import torch
import pytorch_lightning as pl
import pl_bolts
import torchmetrics
import MinkowskiEngine as ME

from src.utils.metric import per_class_iou


@gin.configurable
class LitSegmentationModuleBase(pl.LightningModule):
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
        dist_sync_metric=False,
        lr_eta_min=0.,
    ):
        super(LitSegmentationModuleBase, self).__init__()
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.best_metric_value = -np.inf if best_metric_type == "maximize" else np.inf
        self.metric = torchmetrics.ConfusionMatrix(
            num_classes=num_classes,
            compute_on_step=False,
            dist_sync_on_step=dist_sync_metric
        )

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
        logits = self.model(in_data)
        loss = self.criterion(logits, batch["labels"])
        self.log("train_loss", loss.item(), batch_size=batch["batch_size"], logger=True, prog_bar=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        in_data = self.prepare_input_data(batch)
        logits = self.model(in_data)
        loss = self.criterion(logits, batch["labels"])
        self.log("val_loss", loss.item(), batch_size=batch["batch_size"], logger=True, prog_bar=True)
        pred = logits.argmax(dim=1, keepdim=False)
        mask = batch["labels"] != self.ignore_label
        self.metric(pred[mask], batch["labels"][mask])
        torch.cuda.empty_cache()
        return loss

    def validation_epoch_end(self, outputs):
        confusion_matrix = self.metric.compute().cpu().numpy()
        self.metric.reset()
        ious = per_class_iou(confusion_matrix) * 100
        accs = confusion_matrix.diagonal() / confusion_matrix.sum(1) * 100
        miou = np.nanmean(ious)
        macc = np.nanmean(accs)

        def compare(prev, cur):
            return prev < cur if self.best_metric_type == "maximize" else prev > cur
        
        if compare(self.best_metric_value, miou):
            self.best_metric_value = miou
        self.log("val_best_mIoU", self.best_metric_value, logger=True)
        self.log("val_mIoU", miou, logger=True)
        self.log("val_mAcc", macc, logger=True)

    def prepare_input_data(self, batch):
        raise NotImplementedError


@gin.configurable
class LitSegMinkowskiModule(LitSegmentationModuleBase):
    def prepare_input_data(self, batch):
        in_data = ME.TensorField(
            features=batch["features"],
            coordinates=batch["coordinates"],
            quantization_mode=self.model.QMODE
        )
        return in_data