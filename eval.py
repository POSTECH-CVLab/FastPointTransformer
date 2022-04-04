import argparse

import gin
import torch
import torchmetrics
import MinkowskiEngine as ME
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table

from src.models import get_model
from src.data import get_data_module
from src.utils.metric import per_class_iou


def print_results(classnames, confusion_matrix):
    # results
    ious = per_class_iou(confusion_matrix) * 100
    accs = confusion_matrix.diagonal() / confusion_matrix.sum(1) * 100
    miou = np.nanmean(ious)
    macc = np.nanmean(accs)
    
    # print results
    console = Console()
    table = Table(show_header=True, header_style="bold")

    columns = ["mAcc", "mIoU"]
    num_classes = len(classnames)
    for i in range(num_classes):
        columns.append(classnames[i])
    for col in columns:
        table.add_column(col)
    ious = ious.tolist()
    row = [macc, miou, *ious]
    table.add_row(*[f"{x:.2f}" for x in row])
    console.print(table)


@gin.configurable
def eval(
    checkpoint_path,
    model_name,
    data_module_name,
):
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    ckpt = torch.load(checkpoint_path)

    def remove_prefix(k, prefix):
        return k[len(prefix) :] if k.startswith(prefix) else k

    state_dict = {remove_prefix(k, "model."): v for k, v in ckpt["state_dict"].items()}
    model = get_model(model_name)()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    data_module = get_data_module(data_module_name)()
    data_module.setup("test")
    val_loader = data_module.val_dataloader()

    confmat = torchmetrics.ConfusionMatrix(
        num_classes=data_module.dset_val.NUM_CLASSES, compute_on_step=False
    )
    with torch.inference_mode(mode=True):
        for batch in track(val_loader):
            in_data = ME.TensorField(
                features=batch["features"],
                coordinates=batch["coordinates"],
                quantization_mode=model.QMODE,
                device=device
            )
            pred = model(in_data).cpu()
            mask = batch["labels"] != data_module.dset_val.ignore_label
            confmat(pred[mask], batch["labels"][mask])
            torch.cuda.empty_cache()
    confmat = confmat.compute().numpy()

    cnames = data_module.dset_val.get_classnames()
    print_results(cnames, confmat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("ckpt_path", type=str)
    args = parser.parse_args()

    gin.parse_config_file(args.config)
    eval(args.ckpt_path)