import os.path as osp
from typing import Optional

import gin
import numpy as np
from plyfile import PlyData
from pandas import DataFrame
import torch
import pytorch_lightning as pl

from src.data.collate import CollationFunctionFactory
import src.data.transforms as T

SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),  # No 13
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),  # No 31
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}
VALID_CLASS_LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
VALID_CLASS_NAMES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
                     'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
                     'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                     'bathtub', 'otherfurniture')


def read_ply(filename):
    with open(osp.join(filename), 'rb') as f:
        plydata = PlyData.read(f)
    assert plydata.elements
    data = DataFrame(plydata.elements[0].data).values
    return data


@gin.configurable
class ScanNetDatasetBase(torch.utils.data.Dataset):
    IN_CHANNELS = None
    CLASS_LABELS = None
    SPLIT_FILES = {
        'train': 'scannetv2_train.txt',
        'val': 'scannetv2_val.txt',
        'trainval': 'scannetv2_trainval.txt',
        'test': 'scannetv2_test.txt',
        'overfit': 'scannetv2_overfit.txt'
    }

    def __init__(self, phase, data_root, transform=None, ignore_label=255):
        assert self.IN_CHANNELS is not None
        assert self.CLASS_LABELS is not None
        assert phase in self.SPLIT_FILES.keys()
        super(ScanNetDatasetBase, self).__init__()

        self.phase = phase
        self.data_root = data_root
        self.transform = transform
        self.ignore_label = ignore_label
        self.split_file = self.SPLIT_FILES[phase]
        self.ignore_class_labels = tuple(set(range(41)) - set(self.CLASS_LABELS))
        self.labelmap = self.get_labelmap()
        self.labelmap_inverse = self.get_labelmap_inverse()

        with open(osp.join(self.data_root, 'meta_data', self.split_file), 'r') as f:
            filenames = f.read().splitlines()

        sub_dir = 'test' if phase == 'test' else 'train'
        self.filenames = [
            osp.join(self.data_root, 'scannet_processed', sub_dir, f'{filename}.ply')
            for filename in filenames
        ]

    def __len__(self):
        return len(self.filenames)

    def get_classnames(self):
        classnames = {}
        for class_id in self.CLASS_LABELS:
            classnames[self.labelmap[class_id]] = VALID_CLASS_NAMES[VALID_CLASS_LABELS.index(class_id)]
        return classnames

    def get_colormaps(self):
        colormaps = {}
        for class_id in self.CLASS_LABELS:
            colormaps[self.labelmap[class_id]] = SCANNET_COLOR_MAP[class_id]
        return colormaps

    def get_labelmap(self):
        labelmap = {}
        for k in range(41):
            if k in self.ignore_class_labels:
                labelmap[k] = self.ignore_label
            else:
                labelmap[k] = self.CLASS_LABELS.index(k)
        return labelmap

    def get_labelmap_inverse(self):
        labelmap_inverse = {}
        for k, v in self.labelmap.items():
            labelmap_inverse[v] = self.ignore_label if v == self.ignore_label else k
        return labelmap_inverse


@gin.configurable
class ScanNetRGBDataset(ScanNetDatasetBase):
    IN_CHANNELS = 3
    CLASS_LABELS = VALID_CLASS_LABELS
    NUM_CLASSES = len(VALID_CLASS_LABELS) # 20

    def __getitem__(self, idx):
        data = self._load_data(idx)
        coords, feats, labels = self.get_cfl_from_data(data)
        if self.transform is not None:
            coords, feats, labels = self.transform(coords, feats, labels)
        coords = torch.from_numpy(coords)
        feats = torch.from_numpy(feats)
        labels = torch.from_numpy(labels)
        return coords.float(), feats.float(), labels.long(), None

    def _load_data(self, idx):
        filename = self.filenames[idx]
        data = read_ply(filename)
        return data

    def get_cfl_from_data(self, data):
        xyz, rgb, labels = data[:, :3], data[:, 3:6], data[:, -2]
        labels = np.array([self.labelmap[x] for x in labels])
        return (
            xyz.astype(np.float32),
            rgb.astype(np.float32), 
            labels.astype(np.int64)
        )


@gin.configurable
class ScanNetRGBReconDataset(ScanNetRGBDataset):
    def __getitem__(self, idx):
        data = self._load_data(idx)
        coords, feats, _ = self.get_cfl_from_data(data)
        if self.transform is not None:
            coords, feats, _ = self.transform(coords, feats, None)
        coords = torch.from_numpy(coords)
        feats = torch.from_numpy(feats)
        return coords.float(), feats.float(), feats.clone(), None


@gin.configurable
class ScanNetRGBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        train_batch_size,
        val_batch_size,
        train_num_workers,
        val_num_workers,
        collation_type,
        train_transforms,
        eval_transforms,
    ):
        super(ScanNetRGBDataModule, self).__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.collate_fn = CollationFunctionFactory(collation_type)
        self.train_transforms_ = train_transforms
        self.eval_transforms_ = eval_transforms

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_transforms = []
            if self.train_transforms_ is not None:
                for name in self.train_transforms_:
                    train_transforms.append(getattr(T, name)())
            train_transforms = T.Compose(train_transforms)
            self.dset_train = ScanNetRGBDataset("train", self.data_root, train_transforms)
        eval_transforms = []
        if self.eval_transforms_ is not None:
            for name in self.eval_transforms_:
                eval_transforms.append(getattr(T, name)())
        eval_transforms = T.Compose(eval_transforms)
        self.dset_val = ScanNetRGBDataset("val", self.data_root, eval_transforms)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dset_train, batch_size=self.train_batch_size, shuffle=True, drop_last=False,
            num_workers=self.train_num_workers, collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dset_val, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_num_workers, 
            drop_last=False, collate_fn=self.collate_fn
        )


@gin.configurable
class ScanNetRGBReconDataModule(ScanNetRGBDataModule):
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_transforms = []
            if self.train_transforms_ is not None:
                for name in self.train_transforms_:
                    train_transforms.append(getattr(T, name)())
            train_transforms = T.Compose(train_transforms)
            self.dset_train = ScanNetRGBReconDataset("train", self.data_root, train_transforms)
        eval_transforms = []
        if self.eval_transforms_ is not None:
            for name in self.eval_transforms_:
                eval_transforms.append(getattr(T, name)())
        eval_transforms = T.Compose(eval_transforms)
        self.dset_val = ScanNetRGBReconDataset("val", self.data_root, eval_transforms)


@gin.configurable
class ScanNetRGBDataset_(ScanNetRGBDataset):
    def __getitem__(self, idx):
        data, filename = self._load_data(idx)
        coords, feats, labels = self.get_cfl_from_data(data)
        if self.transform is not None:
            coords, feats, labels = self.transform(coords, feats, labels)
        coords = torch.from_numpy(coords)
        feats = torch.from_numpy(feats)
        labels = torch.from_numpy(labels)
        return coords.float(), feats.float(), labels.long(), filename

    def _load_data(self, idx):
        filename = self.filenames[idx]
        data = read_ply(filename)
        return data, filename