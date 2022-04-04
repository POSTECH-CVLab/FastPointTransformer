import gin
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from src.models.resunet import Res16UNet34C


@gin.configurable
class SPVCNN(Res16UNet34C):
    def network_initialization(self, in_channels, out_channels, D):
        super(SPVCNN, self).network_initialization(in_channels, out_channels, D)
        self.final = ME.MinkowskiLinear(self.PLANES[7] * self.BLOCK.expansion, out_channels)

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                ME.MinkowskiLinear(self.INIT_DIM, self.PLANES[3]),
                ME.MinkowskiBatchNorm(self.PLANES[3]),
                ME.MinkowskiReLU(True)
            ),
            nn.Sequential(
                ME.MinkowskiLinear(self.PLANES[4], self.PLANES[5]),
                ME.MinkowskiBatchNorm(self.PLANES[5]),
                ME.MinkowskiReLU(True)
            ),
            nn.Sequential(
                ME.MinkowskiLinear(self.PLANES[5], self.PLANES[7]),
                ME.MinkowskiBatchNorm(self.PLANES[7]),
                ME.MinkowskiReLU(True)
            ),
        ])
        self.dropout = ME.MinkowskiDropout(0.3, True)

    def voxel_to_point(self, s: ME.SparseTensor, f: ME.TensorField):
        feats, _, out_map, weights = ME.MinkowskiInterpolationFunction().apply(
            s.F, f.C, s.coordinate_key, s.coordinate_manager
        )
        denom  = torch.zeros((len(f),), dtype=feats.dtype, device=feats.device)
        denom.index_add_(0, out_map.long(), weights)
        denom.unsqueeze_(1)
        norm_feats = torch.true_divide(feats, denom + 1e-8)
        return ME.TensorField(
            features=norm_feats,
            coordinate_field_map_key=f.coordinate_field_map_key,
            quantization_mode=f.quantization_mode,
            coordinate_manager=f.coordinate_manager
        )

    def forward(self, x: ME.TensorField):
        x0 = x.sparse()
        x0 = self.relu(self.bn0(self.conv0p1s1(x0)))
        z0 = self.voxel_to_point(x0, x)

        x1 = z0.sparse(coordinate_map_key=x0.coordinate_map_key)
        x1 = self.relu(self.bn1(self.conv1p1s2(x1)))
        x1 = self.block1(x1)

        x2 = self.relu(self.bn2(self.conv2p2s2(x1)))
        x2 = self.block2(x2)

        x3 = self.relu(self.bn3(self.conv3p4s2(x2)))
        x3 = self.block3(x3)

        x4 = self.relu(self.bn4(self.conv4p8s2(x3)))
        x4 = self.block4(x4)

        z1 = self.voxel_to_point(x4, x)
        z1 = z1 + self.point_transforms[0](z0).F

        y1 = z1.sparse(coordinate_map_key=x4.coordinate_map_key)
        y1 = self.dropout(y1)
        y1 = self.relu(self.bntr4(self.convtr4p16s2(y1)))
        y1 = ME.cat(y1, x3)
        y1 = self.block5(y1)

        y2 = self.relu(self.bntr5(self.convtr5p8s2(y1)))
        y2 = ME.cat(y2, x2)
        y2 = self.block6(y2)
        z2 = self.voxel_to_point(y2, x)
        z2 = z2 + self.point_transforms[1](z1).F

        y3 = z2.sparse(coordinate_map_key=x2.coordinate_map_key)
        y3 = self.dropout(y3)
        y3 = self.relu(self.bntr6(self.convtr6p4s2(y3)))
        y3 = ME.cat(y3, x1)
        y3 = self.block7(y3)

        y4 = self.relu(self.bntr7(self.convtr7p2s2(y3)))
        y4 = ME.cat(y4, x0)
        y4 = self.block8(y4)
        z3 = self.voxel_to_point(y4, x)
        z3 = z3 + self.point_transforms[2](z2).F
        return self.final(z3).F