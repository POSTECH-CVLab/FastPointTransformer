import logging

from src.data.scannet_loader import *
from src.data.s3dis_loader import *

ALL_DATA_MODULES = [
    ScanNetRGBDataModule,
    S3DISArea5RGBDataModule,
    ScanNetRGBReconDataModule,
]
ALL_DATASETS = [
    ScanNetRGBDataset,
    S3DISArea5RGBDataset,
    ScanNetRGBReconDataset,
    ScanNetRGBDataset_,
]
data_module_str_mapping = {d.__name__: d for d in ALL_DATA_MODULES}
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def get_data_module(name: str):
    if name not in data_module_str_mapping.keys():
        logging.error(
            f"data_module {name}, does not exists in ".join(
                data_module_str_mapping.keys()
            )
        )
    return data_module_str_mapping[name]


def get_dataset(name: str):
    if name not in dataset_str_mapping.keys():
        logging.error(
            f"dataset {name}, does not exists in ".join(
                dataset_str_mapping.keys()
            )
        )
    return dataset_str_mapping[name]
