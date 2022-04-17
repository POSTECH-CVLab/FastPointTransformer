import os
import errno
import glob

import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement


STANFORD_3D_IN_PATH = '/root/data/s3dis/Stanford3dDataset_v1.2/' # you may need to modify this path.
STANFORD_3D_OUT_PATH = '/root/data/s3dis/s3dis_processed' # you may need to modify this path.
IGNORE_LABEL = 255


def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def save_point_cloud(points_3d, filename, binary=True, with_label=False, verbose=True):
  """Save an RGB point cloud as a PLY file.
  Args:
    points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
        the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
  """
  assert points_3d.ndim == 2
  if with_label:
    assert points_3d.shape[1] == 7
    python_types = (float, float, float, int, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1'), ('label', 'u1')]
  else:
    if points_3d.shape[1] == 3:
      gray_concat = np.tile(np.array([128], dtype=np.uint8), (points_3d.shape[0], 3))
      points_3d = np.hstack((points_3d, gray_concat))
    assert points_3d.shape[1] == 6
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1')]
  if binary:
    # Format into NumPy structured array
    vertices = []
    for row_idx in range(points_3d.shape[0]):
      cur_point = points_3d[row_idx]
      vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
    vertices_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(vertices_array, 'vertex')

    # Write
    PlyData([el]).write(filename)
  else:
    raise NotImplementedError
  if verbose is True:
    print('Saved point cloud to: %s' % filename)


class Stanford3DDatasetConverter:

  CLASSES = [
    'ceiling', 'floor', 'wall', 'beam', 'column',
    'window', 'door', 'chair', 'table', 'bookcase',
    'sofa', 'board', 'clutter'
  ]
  TRAIN_TEXT = 'train'
  VAL_TEXT = 'val'
  TEST_TEXT = 'test'

  @classmethod
  def read_txt(cls, txtfile):
    # Read txt file and parse its content.
    obj_name = txtfile.split('/')[-1]
    if obj_name == 'ceiling_1.txt':
      with open(txtfile, 'r') as f:
        lines = f.readlines()
      for l_i, line in enumerate(lines):
        # https://github.com/zeliu98/CloserLook3D/issues/15
        if '103.0\x100000' in line:
          print(f'Fix bug in {txtfile}')
          print(f'Bug line: {line}')
          lines[l_i] = line.replace('103.0\x100000', '103.000000')
      with open(txtfile, 'w') as f:
        f.writelines(lines)
    try:
      pointcloud = np.loadtxt(txtfile, dtype=np.float32)
    except:
      print('Bug!!!!!!!!!!!!!!!!!!!!!')
      print(obj_name)
      print(txtfile)

    # Load point cloud to named numpy array.
    pointcloud = np.array(pointcloud).astype(np.float32)
    assert pointcloud.shape[1] == 6
    xyz = pointcloud[:, :3].astype(np.float32)
    rgb = pointcloud[:, 3:].astype(np.uint8)
    return xyz, rgb

  @classmethod
  def convert_to_ply(cls, root_path, out_path):
    """Convert Stanford3DDataset to PLY format that is compatible with
    Synthia dataset. Assumes file structure as given by the dataset.
    Outputs the processed PLY files to `STANFORD_3D_OUT_PATH`.
    """

    txtfiles = glob.glob(os.path.join(root_path, '*/*/*.txt'))
    for txtfile in tqdm(txtfiles):
      area_name = txtfile.split('/')[-3]
      file_sp = os.path.normpath(txtfile).split(os.path.sep)
      target_path = os.path.join(out_path, file_sp[-3])
      out_file = os.path.join(target_path, file_sp[-2] + '.ply')

      if os.path.exists(out_file):
        print(out_file, ' exists')
        continue

      annotation, _ = os.path.split(txtfile)
      subclouds = glob.glob(os.path.join(annotation, 'Annotations/*.txt'))
      coords, feats, labels = [], [], []
      for inst, subcloud in enumerate(subclouds):
        # Read ply file and parse its rgb values.
        xyz, rgb = cls.read_txt(subcloud)
        _, annotation_subfile = os.path.split(subcloud)
        clsname = annotation_subfile.split('_')[0]
        # https://github.com/chrischoy/SpatioTemporalSegmentation/blob/4afee296ebe387d9a06fc1b168c4af212a2b4804/lib/datasets/stanford.py#L20
        if clsname == 'stairs':
            print('Ignore stairs')
            clsidx = IGNORE_LABEL
        else:
            clsidx = cls.CLASSES.index(clsname)

        coords.append(xyz)
        feats.append(rgb)
        labels.append(np.full((len(xyz), 1), clsidx, dtype=np.int32))

      if len(coords) == 0:
        print(txtfile, ' has 0 files.')
      else:
        # Concat
        coords = np.concatenate(coords, 0)
        feats = np.concatenate(feats, 0)
        labels = np.concatenate(labels, 0)

        pointcloud = np.concatenate((coords, feats, labels), axis=1)

        # Write ply file.
        mkdir_p(target_path)
        save_point_cloud(pointcloud, out_file, with_label=True, verbose=False)


if __name__ == '__main__':
  Stanford3DDatasetConverter.convert_to_ply(STANFORD_3D_IN_PATH, STANFORD_3D_OUT_PATH)
