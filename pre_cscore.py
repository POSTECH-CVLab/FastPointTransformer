import os
import gc
import argparse
import os.path as osp

import torch
import numpy as np
import pytorch_lightning as pl
import MinkowskiEngine as ME
from tqdm import tqdm

from src.models import get_model
from src.data.scannet_loader import ScanNetRGBDataset_
from src.data.transforms import NormalizeColor, homogeneous_coords
from src.utils.misc import load_from_pl_state_dict


class SimpleConfig:
    def __init__(
        self,
        scannet_path="/root/data/scannetv2",
        ignore_label=255,
        voxel_size=0.1,
        cache_data=False
    ):
        self.scannet_path = scannet_path
        self.voxel_size= voxel_size
        self.ignore_label = ignore_label
        self.cache_data = cache_data
        self.limit_numpoints = -1


class TransformGenerator:
    def __init__(self, voxel_size=0.1, trans_granularity=3, rot_granularity=8, type='trans'):
        assert type in ['trans', 'rot', 'full']
        self.type = type
        self.voxel_size = voxel_size
        trans_step = 1. / trans_granularity
        self.trans_list = [trans_step * i * voxel_size for i in range(trans_granularity)] # 0, 1, 2, 3, ..., granularity - 1
        rot_step = 1. / rot_granularity
        self.rot_list = [rot_step * i * np.pi for i in range(1, 2*rot_granularity)]
        self.transform_list = []
        self.num_trans = 0
        self.num_rot = 0

    def generate_transforms(self):
        self._cleanup()
        if self.type == 'trans':
            self.transform_list.extend(self._get_trans_mtx_list())
        elif self.type == 'rot':
            self.transform_list.extend(self._get_rot_mtx_list())
        else:
            self.transform_list.extend(self._get_trans_mtx_list())
            self.transform_list.extend(self._get_rot_mtx_list())

    def _get_trans_mtx_list(self):
        mtx_list = []
        for delta_x in self.trans_list:
            for delta_y in self.trans_list:
                for delta_z in self.trans_list:
                    if delta_x == 0 and delta_y == 0 and delta_z == 0:
                        continue
                    T = np.eye(4)
                    T[0, 3] = delta_x
                    T[1, 3] = delta_y
                    T[2, 3] = delta_z
                    mtx_list.append(T)
                    self.num_trans += 1
        return mtx_list

    def _get_rot_mtx_list(self):
        mtx_list = []
        for theta in self.rot_list:
            T = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0, 0],
                    [np.sin(theta), np.cos(theta), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ]
            )
            mtx_list.append(T)
            self.num_rot += 1
        return mtx_list

    def _cleanup(self):
        self.num_trans = 0
        self.num_rot = 0
        self.transform_list = []


def save_outputs(args, ckpt, model, name, tmatrix):
    output_dir = osp.join(args.out_dir, args.model_name + f"-voxel={args.voxel_size}")
    if args.postfix is not None:
        output_dir = output_dir + args.postfix
    transform = NormalizeColor()
    dset = ScanNetRGBDataset_("val", args.scannet_path, transform)
    num_samples = len(dset)
    output_name = osp.join(output_dir, name)
    assert not osp.isdir(output_name)
    os.makedirs(output_name, exist_ok=True)
    
    print(f'>>> Saving predictions for {num_samples} val samples...')
    with torch.inference_mode(mode=True):
        for coords_, feats, labels, fname in dset:
            if tmatrix is not None:
                np.save(osp.join(output_name, "tmatrix.npy"), tmatrix)
                coords = torch.from_numpy(homogeneous_coords(coords_.numpy()) @ tmatrix.T)[:, :3]
            else:
                coords = coords_
            coords, feats = ME.utils.sparse_collate(
                [coords / args.voxel_size],
                [feats],
                dtype=torch.float32
            )
            in_field = ME.TensorField(
                features=feats,
                coordinates=coords,
                quantization_mode=model.QMODE,
                device=device
            )
            pred = model(in_field).argmax(dim=1, keepdim=False).cpu().numpy()
            assert len(pred) == len(labels)
            pred[np.where(labels.numpy() == 255)] = 255 # ignore labels
            scene_id = fname.split('/')[-1].split('.')[0]
            np.save(osp.join(output_name, f'{scene_id}.npy'), pred)
            gc.collect()
            torch.cuda.empty_cache()
    print('    done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", type=str)
    parser.add_argument("-m", "--model_name", type=str, choices=["mink", "fpt"])
    parser.add_argument("-v", "--voxel_size", type=float, choices=[0.1, 0.05, 0.02])
    parser.add_argument("--scannet_path", type=str, default="/root/data/scannetv2")
    parser.add_argument("--type", type=str, default="full", choices=["full"])
    parser.add_argument("--out_dir", type=str, default="/root/data/cvpr2022/consistency_outputs")
    parser.add_argument("-p", "--postfix", type=str, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f">>> Loading the checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt)
    pl.seed_everything(7777)
    print("    done!")

    print(">>> Loading the model...")
    if args.model_name == "mink":
        model = get_model("Res16UNet34C")(ScanNetRGBDataset_.IN_CHANNELS, ScanNetRGBDataset_.NUM_CLASSES)
    else:
        model = get_model("FastPointTransformer")(ScanNetRGBDataset_.IN_CHANNELS, ScanNetRGBDataset_.NUM_CLASSES)
    model = load_from_pl_state_dict(model, ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    print("    done!")

    print(">>> Generating rigid transformations...")
    tgenerator = TransformGenerator(voxel_size=args.voxel_size, type=args.type)
    tgenerator.generate_transforms()
    print(f'    {len(tgenerator.transform_list)} rigid transformations generated!')

    print(f">>> Evaluating the model...")
    save_outputs(args, ckpt, model, "reference", None)
    for t_idx, tmatrix in enumerate(tqdm(tgenerator.transform_list)):
        print(tmatrix)
        save_outputs(args, ckpt, model, f"transform{t_idx}", tmatrix)
    print("    done!")