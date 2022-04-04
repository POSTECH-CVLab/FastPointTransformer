import logging
import random

import gin
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
from scipy.linalg import expm, norm
import torch
import MinkowskiEngine as ME


def homogeneous_coords(coords):
  assert isinstance(coords, np.ndarray) and coords.shape[1] == 3
  return np.concatenate([coords, np.ones((len(coords), 1))], axis=1)


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


# A sparse tensor consists of coordinates and associated features.
# You must apply augmentation to both.
# In 2D, flip, shear, scale, and rotation of images are coordinate transformation
# color jitter, hue, etc., are feature transformations
##############################
# Coordinate transformations
##############################
@gin.configurable
class ElasticDistortion:
    def __init__(self, distortion_params=[(4, 16), (8, 24)], application_ratio=0.9):
        self.application_ratio = application_ratio
        self.distortion_params = distortion_params
        logging.info(
            f"{self.__class__.__name__} distortion_params:{distortion_params} with application_ratio:{application_ratio}"
        )

    def elastic_distortion(self, coords, feats, labels, granularity, magnitude):
        """Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=0, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords, feats, labels

    def __call__(self, coords, feats, labels):
        if self.distortion_params is not None:
            if random.random() < self.application_ratio:
                for granularity, magnitude in self.distortion_params:
                    coords, feats, labels = self.elastic_distortion(
                        coords, feats, labels, granularity, magnitude
                    )
        return coords, feats, labels


# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


@gin.configurable
class RandomRotation(object):
    def __init__(self, upright_axis="z", axis_std=0.01, application_ratio=0.9):
        self.upright_axis = {"x": 0, "y": 1, "z": 2}[upright_axis.lower()]
        self.D = 3
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])
        self.application_ratio = application_ratio
        self.axis_std = axis_std
        logging.info(
            f"{self.__class__.__name__} upright_axis:{upright_axis}, axis_std:{axis_std} with application_ratio:{application_ratio}"
        )

    def __call__(self, coords, feats, labels):
        if random.random() < self.application_ratio:
            axis = self.axis_std * np.random.randn(3)
            axis[self.upright_axis] += 1
            angle = random.random() * 2 * np.pi
            coords = coords @ M(axis, angle)
        return coords, feats, labels


@gin.configurable
class RandomTranslation(object):
    def __init__(
        self,
        max_translation=3,
        application_ratio=0.9,
    ):
        self.max_translation = max_translation
        self.application_ratio = application_ratio
        logging.info(
            f"{self.__class__.__name__} max_translation:{max_translation} with application_ratio:{application_ratio}"
        )

    def __call__(self, coords, feats, labels):
        if random.random() < self.application_ratio:
            coords += 2 * (np.random.rand(1, 3) - 0.5) * self.max_translation
        return coords, feats, labels


@gin.configurable
class RandomScale(object):
    def __init__(self, scale_ratio=0.1, application_ratio=0.9):
        self.scale_ratio = scale_ratio
        self.application_ratio = application_ratio
        logging.info(f"{self.__class__.__name__}(scale_ratio={scale_ratio})")

    def __call__(self, coords, feats, labels):
        if random.random() < self.application_ratio:
            coords = coords * np.random.uniform(
                low=1 - self.scale_ratio, high=1 + self.scale_ratio
            )
        return coords, feats, labels


@gin.configurable
class RandomCrop(object):
    def __init__(self, x, y, z, application_ratio=1, min_cardinality=100, max_retries=10):
        assert x > 0
        assert y > 0
        assert z > 0
        self.application_ratio = application_ratio
        self.max_size = np.array([[x, y, z]])
        self.min_cardinality = min_cardinality
        self.max_retries = max_retries
        logging.info(f"{self.__class__.__name__} with size {self.max_size}")

    def __call__(self, coords: np.array, feats, labels):
        if random.random() > self.application_ratio:
            return coords, feats, labels

        norm_coords = coords - coords.min(0, keepdims=True)
        max_coords = norm_coords.max(0, keepdims=True)
        # start range
        coord_range = max_coords - self.max_size
        coord_range = np.clip(coord_range, a_min=0, a_max=float("inf"))
        # If crop size is larger than the coordinates, return orig
        if np.prod(coord_range == 0):
            return coords, feats, labels

        # sample crop start point
        valid = False
        retries = 0
        while not valid:
            min_box = np.random.rand(1, 3) * coord_range
            max_box = min_box + self.max_size
            sel = np.logical_and(
                np.prod(norm_coords > min_box, 1), np.prod(norm_coords < max_box, 1)
            )
            if np.sum(sel) > self.min_cardinality:
                valid = True
            retries += 1
            if retries % 2 == 0:
                logging.warn(f"RandomCrop retries: {retries}. crop_range={coord_range}")
            if retries >= self.max_retries:
                break

        if valid:
            return (
                coords[sel],
                feats if feats is None else feats[sel],
                labels if labels is None else labels[sel],
            )
        return coords, feats, labels


@gin.configurable
class RandomAffine(object):
    def __init__(
        self,
        upright_axis="z",
        axis_std=0.1,
        scale_range=0.2,
        affine_range=0.1,
        application_ratio=0.9,
    ):
        self.upright_axis = {"x": 0, "y": 1, "z": 2}[upright_axis.lower()]
        self.D = 3
        self.scale_range = scale_range
        self.affine_range = affine_range
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])
        self.application_ratio = application_ratio
        self.axis_std = axis_std
        logging.info(
            f"{self.__class__.__name__} upright_axis:{upright_axis}, axis_std:{axis_std}, scale_range:{scale_range}, affine_range:{affine_range} with application_ratio:{application_ratio}"
        )

    def __call__(self, coords, feats, labels):
        if random.random() < self.application_ratio:
            axis = self.axis_std * np.random.randn(3)
            axis[self.upright_axis] += 1
            angle = 2 * (random.random() - 0.5) * np.pi
            T = M(axis, angle) @ (
                np.diag(2 * (np.random.rand(3) - 0.5) * self.scale_range + 1)
                + 2 * (np.random.rand(3, 3) - 0.5) * self.affine_range
            )
            coords = coords @ T
        return coords, feats, labels


@gin.configurable
class RandomHorizontalFlip(object):
    def __init__(self, upright_axis="z", application_ratio=0.9):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.D = 3
        self.upright_axis = {"x": 0, "y": 1, "z": 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])
        self.application_ratio = application_ratio
        logging.info(
            f"{self.__class__.__name__} upright_axis:{upright_axis} with application_ratio:{application_ratio}"
        )

    def __call__(self, coords, feats, labels):
        if random.random() < self.application_ratio:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = np.max(coords[:, curr_ax])
                    coords[:, curr_ax] = coord_max - coords[:, curr_ax]
        return coords, feats, labels


@gin.configurable
class CoordinateDropout(object):
    def __init__(self, dropout_ratio=0.2, application_ratio=0.2):
        self.dropout_ratio = dropout_ratio
        self.application_ratio = application_ratio
        logging.info(
            f"{self.__class__.__name__} dropout:{dropout_ratio} with application_ratio:{application_ratio}"
        )

    def __call__(self, coords, feats, labels):
        if random.random() < self.application_ratio:
            N = len(coords)
            inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
            return (
                coords[inds],
                feats if feats is None else feats[inds],
                labels if labels is None else labels[inds],
            )
        return coords, feats, labels


@gin.configurable
class CoordinateJitter(object):
    def __init__(self, jitter_std=0.5, application_ratio=0.7):
        self.jitter_std = jitter_std
        self.application_ratio = application_ratio
        logging.info(
            f"{self.__class__.__name__} jitter_std:{jitter_std} with application_ratio:{application_ratio}"
        )

    def __call__(self, coords, feats, labels):
        if random.random() < self.application_ratio:
            N = len(coords)
            coords += (2 * self.jitter_std) * (np.random.rand(N, 3) - 0.5)
        return coords, feats, labels


@gin.configurable
class CoordinateUniformTranslation:
    def __init__(self, max_translation=0.2):
        self.max_translation = max_translation

    def __call__(self, coords, feats, labels):
        if self.max_translation > 0:
            coords += np.random.uniform(
                low=-self.max_translation, high=self.max_translation, size=[1, 3]
            )
        return coords, feats, labels


@gin.configurable
class RegionDropout(object):
    def __init__(
        self,
        box_center_range=[100, 100, 10],
        max_region_size=[300, 300, 300],
        min_region_size=[100, 100, 100],
        application_ratio=0.3,
    ):
        self.max_region_size = np.array(max_region_size)
        self.min_region_size = np.array(min_region_size)
        self.box_range = self.max_region_size - self.min_region_size
        self.box_center_range = np.array([box_center_range])
        self.application_ratio = application_ratio
        logging.info(
            f"{self.__class__.__name__} max_region_size:{max_region_size} min_region_size:{min_region_size} box_center_range:{box_center_range} with application_ratio:{application_ratio}"
        )

    def __call__(self, coords, feats, labels):
        if random.random() < self.application_ratio:
            while True:
                box_center = self.box_center_range * (
                    np.random.rand(1, 3) - 0.5
                ) * 2 + np.mean(coords, axis=0, keepdims=True)
                box_size = self.box_range * np.random.rand(1, 3)
                min_xyz = box_center - box_size / 2
                max_xyz = box_center + box_size / 2
                sel = np.logical_not(
                    np.prod(coords < max_xyz, axis=1)
                    * np.prod(coords > min_xyz, axis=1)
                )
                if sel.sum() > len(coords) * 0.5:
                    break
            return coords[sel], feats[sel], labels[sel]
        return coords, feats, labels


@gin.configurable
class DimensionlessCoordinates(object):
    def __init__(self, voxel_size=0.02):
        self.voxel_size = voxel_size
        logging.info(f"{self.__class__.__name__} with voxel_size:{voxel_size}")

    def __call__(self, coords, feats, labels):
        return coords / self.voxel_size, feats, labels


@gin.configurable
class PerlinNoise:
    def __init__(
        self, noise_params=[(4, 4), (16, 16)], application_ratio=0.9, device="cpu"
    ):
        self.application_ratio = application_ratio
        self.noise_params = noise_params
        logging.info(
            f"{self.__class__.__name__} noise_params:{noise_params} with application_ratio:{application_ratio}"
        )
        self.interp = ME.MinkowskiInterpolation()
        self.corners = torch.Tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [0, 1, 1],
                [1, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
            ]
        )
        self.smooth = ME.MinkowskiConvolution(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            bias=False,
            dimension=3,
        )
        self.smooth.kernel[:] = 1 / 27
        if device is None and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.smooth = self.smooth.to(self.device)
        self.corners = self.corners.to(self.device)

    def perlin_noise(self, coordinates, noise_quantization_size, noise_std):
        aug_coordinates = coordinates.reshape(-1, 1, 3) + (
            self.corners * noise_quantization_size
        ).reshape(1, 8, 3)
        bcoords = ME.utils.batched_coordinates(
            [aug_coordinates.reshape(-1, 3) / noise_quantization_size],
            dtype=torch.float32,
        )
        noise_tensor = ME.SparseTensor(
            features=torch.randn((len(bcoords), 3), device=self.device),
            coordinates=bcoords,
            device=self.device,
        )
        noise_tensor = self.smooth(noise_tensor)
        interp_noise = self.interp(
            noise_tensor,
            ME.utils.batched_coordinates(
                [coordinates / noise_quantization_size],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        return coordinates + noise_std * interp_noise

    def __call__(self, coords, feats, labels):
        if self.noise_params is not None:
            if random.random() < self.application_ratio:
                coords = torch.from_numpy(coords).to(self.device)
                with torch.no_grad():
                    for quantization_size, noise_std in self.noise_params:
                        coords = self.perlin_noise(coords, quantization_size, noise_std)
                coords = coords.cpu().numpy()
        return coords, feats, labels


##############################
# Feature transformations
##############################
@gin.configurable
class ChromaticTranslation(object):
    """Add random color to the image, input must be an array in [0,255] or a PIL image"""

    def __init__(self, translation_range_ratio=1e-1, application_ratio=0.9):
        """
        trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
        """
        self.trans_range_ratio = translation_range_ratio
        self.application_ratio = application_ratio
        logging.info(
            f"{self.__class__.__name__} with translation_range_ratio:{translation_range_ratio} with application_ratio:{application_ratio}"
        )

    def __call__(self, coords, feats, labels):
        if random.random() < self.application_ratio:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
            feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
        return coords, feats, labels


@gin.configurable
class ChromaticJitter(object):
    def __init__(self, std=0.01, application_ratio=0.9):
        self.std = std
        self.application_ratio = application_ratio
        logging.info(
            f"{self.__class__.__name__} with std:{std} with application_ratio:{application_ratio}"
        )

    def __call__(self, coords, feats, labels):
        if random.random() < self.application_ratio:
            noise = np.random.randn(feats.shape[0], 3)
            noise *= self.std * 255
            feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
        return coords, feats, labels


@gin.configurable
class ChromaticAutoContrast(object):
    def __init__(
        self, randomize_blend_factor=True, blend_factor=0.5, application_ratio=0.2
    ):
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor
        self.application_ratio = application_ratio
        logging.info(
            f"{self.__class__.__name__} with randomize_blend_factor:{randomize_blend_factor}, blend_factor:{blend_factor} with application_ratio:{application_ratio}"
        )

    def __call__(self, coords, feats, labels):
        if random.random() < self.application_ratio:
            # mean = np.mean(feats, 0, keepdims=True)
            # std = np.std(feats, 0, keepdims=True)
            # lo = mean - std
            # hi = mean + std
            lo = feats[:, :3].min(0, keepdims=True)
            hi = feats[:, :3].max(0, keepdims=True)
            assert hi.max() > 1, f"invalid color value. Color is supposed to be [0-255]"

            if np.prod(hi - lo):
                scale = 255 / (hi - lo)
                contrast_feats = (feats[:, :3] - lo) * scale
                blend_factor = (
                    random.random() if self.randomize_blend_factor else self.blend_factor
                )
                feats[:, :3] = (1 - blend_factor) * feats + blend_factor * contrast_feats
        return coords, feats, labels


@gin.configurable
class NormalizeColor(object):
    def __init__(self, mean=[128, 128, 128], std=[256, 256, 256], pre_norm=False):
        self.mean = np.array([mean], dtype=np.float32)
        self.std = np.array([std], dtype=np.float32)
        self.pre_norm = pre_norm
        logging.info(f"{self.__class__.__name__} mean:{mean} std:{std}")

    def __call__(self, coords, feats, labels):
        if self.pre_norm:
            feats = feats / 255.
        return coords, (feats - self.mean) / self.std, labels


@gin.configurable
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype("float")
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc
        )
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype("uint8")
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype("uint8")

    def __init__(self, hue_max, saturation_max):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, coords, feats, labels):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
        hue_val = (random.random() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (random.random() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feats[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
        return coords, feats, labels