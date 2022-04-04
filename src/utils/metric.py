import torch
import numpy as np


def fast_hist(pred, target, num_classes, ignore_label=255):
  mask = (target != ignore_label) & (target < num_classes)
  return np.bincount(num_classes * target[mask].astype(int) + pred[mask],
                     minlength=num_classes**2).reshape(num_classes, num_classes)


def per_class_iou(hist):
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def cal_psnr_from_mse(mse, max_value=1.):
  mse = mse.item() if isinstance(mse, torch.Tensor) else mse
  psnr = 20 * np.log10(max_value) - 10 * np.log10(mse)
  return psnr