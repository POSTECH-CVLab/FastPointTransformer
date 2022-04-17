import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import numpy as np
from plyfile import PlyData, PlyElement


SCANNET_RAW_PATH = Path('/root/data/scannetv2_raw') # you may need to modify this path.
SCANNET_OUT_PATH = Path('/root/data/scannet_processed') # you may need to modify this path.
TRAIN_DEST = 'train'
TEST_DEST = 'test'
SUBSETS = {TRAIN_DEST: 'scans', TEST_DEST: 'scans_test'}
POINTCLOUD_FILE = '_vh_clean_2.ply'
BUGS = {
    'train/scene0270_00.ply': 50,
    'train/scene0270_02.ply': 50,
    'train/scene0384_00.ply': 149,
} # https://github.com/ScanNet/ScanNet/issues/20


def read_plyfile(path):
    with open(path, 'rb') as f:
        data = PlyData.read(f)
    if data.elements:
        return pd.DataFrame(data.elements[0].data).values


def save_point_cloud(points_3d, filename, verbose=True):
    assert points_3d.ndim == 2
    assert points_3d.shape[1] == 8 # x, y, z, r, g, b, semantic_label, instance_label

    python_types = (float, float, float, int, int, int, int, int)
    npy_types = [
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
        ('semantic_label', 'u1'),
        ('instance_label', 'u1')
    ]

    vertices = []
    for row_idx in range(points_3d.shape[0]):
        cur_point = points_3d[row_idx]
        vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
    vertices_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(vertices_array, 'vertex')
    # Write
    PlyData([el]).write(filename)

    if verbose:
        print(f'Saved point cloud to: {filename}')


def handle_process(paths):
    f = paths[0]
    phase_out_path = paths[1]
    out_f = phase_out_path / (f.name[:-len(POINTCLOUD_FILE)] + f.suffix)
    pointcloud = read_plyfile(f)
    num_points = pointcloud.shape[0]
    # Make sure alpha value is meaningless.
    assert np.unique(pointcloud[:, -1]).size == 1
    # Load label file.
    segment_f = f.with_suffix('.0.010000.segs.json')
    segment_group_f = (f.parent / f.name[:-len(POINTCLOUD_FILE)]).with_suffix('.aggregation.json')
    semantic_f = f.parent / (f.stem + '.labels' + f.suffix)

    if semantic_f.is_file():
        semantic_label = read_plyfile(semantic_f)
        # Sanity check that the pointcloud and its label has same vertices.
        assert pointcloud.shape[0] == semantic_label.shape[0]
        assert np.allclose(pointcloud[:, :3], semantic_label[:, :3])
        semantic_label = semantic_label[:, -1]
        # Load instance label
        with open(segment_f) as f:
            segment = np.array(json.load(f)['segIndices'])
        with open(segment_group_f) as f:
            segment_groups = json.load(f)['segGroups']
        assert segment.size == num_points
        instance_label = np.zeros(num_points)
        for group_idx, segment_group in enumerate(segment_groups):
            for segment_idx in segment_group['segments']:
                instance_label[segment == segment_idx] = group_idx + 1
    else:  # Label may not exist in test case.
        semantic_label = np.zeros(num_points)
        instance_label = np.zeros(num_points)

    processed = np.hstack((pointcloud[:, :6], semantic_label[:, None], instance_label[:, None]))
    save_point_cloud(processed, out_f, verbose=False)


def main():
    path_list = []
    for out_path, in_path in SUBSETS.items():
        phase_out_path = SCANNET_OUT_PATH / out_path
        phase_out_path.mkdir(parents=True, exist_ok=True)
        for f in (SCANNET_RAW_PATH / in_path).glob('*/*' + POINTCLOUD_FILE):
            path_list.append([f, phase_out_path])

    pool = ProcessPoolExecutor(max_workers=20)
    result = list(pool.map(handle_process, path_list))

    # Fix bug in the data.
    for files, bug_index in BUGS.items():
        print(files)

        for f in SCANNET_OUT_PATH.glob(files):
            pointcloud = read_plyfile(f)
            bug_mask = pointcloud[:, -2] == bug_index
            print(f'Fixing {f} bugged label {bug_index} x {bug_mask.sum()}...')
            pointcloud[bug_mask, -2] = 0
            save_point_cloud(pointcloud, f, verbose=False)


if __name__ == '__main__':
    print('Preprocessing ScanNetV2 dataset...')
    main()
