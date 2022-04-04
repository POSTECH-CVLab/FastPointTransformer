import gin
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock

from src.models.resnet import ResNetBase
from src.models.common import downsample_points, downsample_embeddings


@gin.configurable
class Res16UNetBase(ResNetBase):
    INIT_DIM = 32
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)

    def __init__(self, in_channels, out_channels, D=3):
        super(Res16UNetBase, self).__init__(in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = self.LAYER(in_channels, self.inplanes, kernel_size=5, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = self.LAYER(self.inplanes,
                                    self.inplanes,
                                    kernel_size=2,
                                    stride=2,
                                    dimension=D)  # pooling
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = self.LAYER(self.inplanes,
                                    self.inplanes,
                                    kernel_size=2,
                                    stride=2,
                                    dimension=D)  # pooling
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = self.LAYER(self.inplanes,
                                    self.inplanes,
                                    kernel_size=2,
                                    stride=2,
                                    dimension=D)  # pooling
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = self.LAYER(self.inplanes,
                                    self.inplanes,
                                    kernel_size=2,
                                    stride=2,
                                    dimension=D)  # pooling
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(self.inplanes,
                                                            self.PLANES[4],
                                                            kernel_size=2,
                                                            stride=2,
                                                            dimension=D)  # unpooling
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.inplanes = self.PLANES[
            4] + self.PLANES[2] * self.BLOCK.expansion  # concatenated dimension
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(self.inplanes,
                                                            self.PLANES[5],
                                                            kernel_size=2,
                                                            stride=2,
                                                            dimension=D)  # unpooling
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.inplanes = self.PLANES[
            5] + self.PLANES[1] * self.BLOCK.expansion  # concatenated dimension
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(self.inplanes,
                                                            self.PLANES[6],
                                                            kernel_size=2,
                                                            stride=2,
                                                            dimension=D)  # unpooling
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.inplanes = self.PLANES[
            6] + self.PLANES[0] * self.BLOCK.expansion  # concatenated dimension
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(self.inplanes,
                                                            self.PLANES[7],
                                                            kernel_size=2,
                                                            stride=2,
                                                            dimension=D)  # unpooling
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.inplanes = self.PLANES[7] + self.INIT_DIM  # concatenated dimension
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion,
                                            out_channels,
                                            kernel_size=1,
                                            stride=1,
                                            bias=True,
                                            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def voxelize(self, x: ME.TensorField):
        raise NotImplementedError()

    def devoxelize(self, out: ME.SparseTensor, x: ME.TensorField, emb: torch.Tensor):
        raise NotImplementedError()

    def forward(self, x: ME.TensorField):
        out, emb = self.voxelize(x)
        out_p1 = self.relu(self.bn0(self.conv0p1s1(out)))

        out = self.relu(self.bn1(self.conv1p1s2(out_p1)))
        out_p2 = self.block1(out)

        out = self.relu(self.bn2(self.conv2p2s2(out_p2)))
        out_p4 = self.block2(out)

        out = self.relu(self.bn3(self.conv3p4s2(out_p4)))
        out_p8 = self.block3(out)

        out = self.relu(self.bn4(self.conv4p8s2(out_p8)))
        out = self.block4(out)

        out = self.relu(self.bntr4(self.convtr4p16s2(out)))
        out = ME.cat(out, out_p8)
        out = self.block5(out)

        out = self.relu(self.bntr5(self.convtr5p8s2(out)))
        out = ME.cat(out, out_p4)
        out = self.block6(out)

        out = self.relu(self.bntr6(self.convtr6p4s2(out)))
        out = ME.cat(out, out_p2)
        out = self.block7(out)

        out = self.relu(self.bntr7(self.convtr7p2s2(out)))
        out = ME.cat(out, out_p1)
        out = self.block8(out)
        return self.devoxelize(out, x, emb)


@gin.configurable
class Res16UNet34C(Res16UNetBase): # MinkowskiNet42
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    
    def voxelize(self, x: ME.TensorField):
        return x.sparse(), None
    
    def devoxelize(self, out: ME.SparseTensor, x: ME.TensorField, emb: torch.Tensor):
        return self.final(out).slice(x).F


@gin.configurable
class Res16UNet34CSmall(Res16UNet34C):
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


@gin.configurable
class Res16UNet34CSmaller(Res16UNet34C):
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


@gin.configurable
class CenRes16UNet34CSmaller(Res16UNet34CSmaller):
    ENC_DIM = 32

    def network_initialization(self, in_channels, out_channels, D=3):
        super(CenRes16UNet34CSmaller, self).network_initialization(in_channels, out_channels, D)
        self.conv0p1s1 = self.LAYER(in_channels + self.ENC_DIM, self.INIT_DIM, kernel_size=5, dimension=D)
        self.enc_mlp = nn.Sequential(
            nn.Linear(D, self.ENC_DIM, bias=False),
            nn.BatchNorm1d(self.ENC_DIM),
            nn.Tanh(),
            nn.Linear(self.ENC_DIM, self.ENC_DIM, bias=False),
            nn.BatchNorm1d(self.ENC_DIM),
            nn.Tanh()
        )
        self.final = nn.Sequential(
            nn.Linear(self.PLANES[7] + self.ENC_DIM, self.PLANES[7], bias=False),
            nn.BatchNorm1d(self.PLANES[7]),
            nn.ReLU(inplace=True),
            nn.Linear(self.PLANES[7], out_channels)
        )

    @torch.no_grad()
    def normalize_points(self, points, centroids, tensor_map):
        tensor_map = tensor_map if tensor_map.dtype == torch.int64 else tensor_map.long()
        norm_points = points - centroids[tensor_map]
        return norm_points

    def voxelize(self, x: ME.TensorField):
        cm = x.coordinate_manager
        points = x.C[:, 1:]
        out = x.sparse()
        size = torch.Size([len(out), len(x)])
        tensor_map, field_map = cm.field_to_sparse_map(x.coordinate_key, out.coordinate_key)
        points_p1, _ = downsample_points(points, tensor_map, field_map, size)
        norm_points = self.normalize_points(points, points_p1, tensor_map)

        emb = self.enc_mlp(norm_points)
        down_emb = downsample_embeddings(emb, tensor_map, size, mode="avg")
        out = ME.SparseTensor(
            torch.cat([out.F, down_emb], dim=1),
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=cm
        )
        return out, emb

    def devoxelize(self, out: ME.SparseTensor, x: ME.TensorField, emb: torch.Tensor):
        out = self.final(torch.cat([out.slice(x).F, emb], dim=1))
        return out