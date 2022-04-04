import logging

import gin
import MinkowskiEngine as ME


@gin.configurable
class CollationFunctionFactory:
    def __init__(self, collation_type="collate_default"):
        if collation_type == "collate_default":
            self.collation_fn = self.collate_default
        elif collation_type == "collate_minkowski":
            self.collation_fn = self.collate_minkowski
        else:
            raise ValueError(f"collation_type {collation_type} not found")

    def __call__(self, list_data):
        return self.collation_fn(list_data)

    def collate_default(self, list_data):
        return list_data

    def collate_minkowski(self, list_data):
        B = len(list_data)
        list_data = [data for data in list_data if data is not None]
        if B != len(list_data):
            logging.info(f"Retain {len(list_data)} from {B} data.")
        if len(list_data) == 0:
            raise ValueError("No data in the batch")

        coords, feats, labels, extra_packages = list(zip(*list_data))
        row_splits = [c.shape[0] for c in coords]
        coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(
            coords, feats, labels, dtype=coords[0].dtype
        )
        return {
            "coordinates": coords_batch,
            "features": feats_batch,
            "labels": labels_batch,
            "row_splits": row_splits,
            "batch_size": B,
            "extra_packages": extra_packages,
        }