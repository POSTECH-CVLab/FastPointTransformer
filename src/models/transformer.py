import math

import gin
import torch
import torch.nn as nn
from torch_scatter import scatter_softmax
import MinkowskiEngine as ME

from src.models.transformer_base import LocalSelfAttentionBase
from src.models.resunet import Res16UNetBase
import src.cuda_ops.functions.sparse_ops as ops


@gin.configurable
class MultiHeadSelfAttention(LocalSelfAttentionBase):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
        dilation=1,
        num_heads=8,
        dimension=3
    ):
        out_channels = in_channels if out_channels is None else out_channels
        assert in_channels % num_heads == 0
        assert kernel_size % 2 == 1
        assert stride == 1, "Currently, this layer only supports stride == 1"
        assert dilation == 1,"Currently, this layer only supports dilation == 1"
        super(MultiHeadSelfAttention, self).__init__(kernel_size, stride, dilation, dimension)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.num_heads = num_heads
        self.attn_channels = out_channels // num_heads

        self.to_query = nn.Linear(in_channels, out_channels)
        self.to_key = nn.Linear(in_channels, out_channels)
        self.to_value = nn.Linear(in_channels, out_channels)
        self.to_out = nn.Linear(out_channels, out_channels)
        self.to_pos_enc = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.ReLU(inplace=True),
            nn.Linear(dimension, out_channels)
        )
        self.rel_pos = nn.Parameter(self.generate_rel_pos(), requires_grad=False) # query - key

    def generate_rel_pos(self):
        # note that MinkowskiEngine uses key - query but we need query - key
        rel_pos = []
        pos_max = self.kernel_size // 2
        for pos_z in reversed(range(-pos_max, pos_max + 1)):
            for pos_y in reversed(range(-pos_max, pos_max + 1)):
                for pos_x in reversed(range(-pos_max, pos_max + 1)):
                    rel_pos.append([pos_x, pos_y, pos_z])
        return torch.FloatTensor(rel_pos)

    def forward(self, stensor):
        dtype = stensor._F.dtype
        device = stensor._F.device

        # query, key, value, and relative positional encoding
        in_F = stensor._F
        q = self.to_query(in_F).view(-1, self.num_heads, self.attn_channels).contiguous()
        k = self.to_key(in_F).view(-1, self.num_heads, self.attn_channels).contiguous()
        v = self.to_value(in_F).view(-1, self.num_heads, self.attn_channels).contiguous()
        r = self.to_pos_enc(self.rel_pos).view(-1, self.num_heads, self.attn_channels).contiguous()

        # key-query map
        kernel_map, out_key = self.get_kernel_map_and_out_key(stensor)
        kq_map = self.key_query_map_from_kernel_map(kernel_map)

        # attention weights with softmax normalization
        attn = torch.zeros((kq_map.shape[1], self.num_heads), dtype=dtype, device=device)
        attn = ops.dot_product_with_key_cuda(q, k, r, attn, kq_map)
        attn = attn / math.sqrt(self.attn_channels)
        attn = scatter_softmax(attn, kq_map[1].long().unsqueeze(1), dim=0)

        # aggregation & the output
        out_F = torch.zeros((len(q), self.num_heads, self.attn_channels),
                            dtype=dtype,
                            device=device)
        kq_indices = self.key_query_indices_from_key_query_map(kq_map)
        out_F = ops.scalar_attention_cuda(attn, v, out_F, kq_indices)
        out_F = self.to_out(out_F.view(-1, self.out_channels).contiguous())
        return ME.SparseTensor(out_F,
                               coordinate_map_key=out_key,
                               coordinate_manager=stensor.coordinate_manager)


@gin.configurable
class MultiHeadSelfAttentionBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels=None,
        stride=1,
        dilation=1,
        downsample=None,
        num_heads=8,
        dimension=3,
    ):
        out_channels = in_channels if out_channels is None else out_channels
        assert downsample is None
        super(MultiHeadSelfAttentionBlock, self).__init__()

        self.norm1 = ME.MinkowskiBatchNorm(in_channels)
        self.mhsa = MultiHeadSelfAttention(
            in_channels, out_channels, stride=stride, dilation=dilation, num_heads=num_heads, dimension=dimension
        )
        self.norm2 = ME.MinkowskiBatchNorm(out_channels)
        self.mlp = nn.Sequential(
            ME.MinkowskiLinear(out_channels, out_channels, bias=False),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(out_channels, out_channels)
        )

    def forward(self, stensor):
        out_ = self.mhsa(stensor)
        out_ = self.norm1(out_)
        out_ += stensor
        out = self.mlp(out_)
        out = self.norm2(out)
        out += out_
        return out


@gin.configurable
class TransformerBase(Res16UNetBase):
    LAYER = MultiHeadSelfAttention

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.mlp = nn.Sequential(
            ME.MinkowskiLinear(in_channels, self.inplanes, bias=False),
            ME.MinkowskiBatchNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True)
        )
        self.attn1p1 = self.LAYER(self.inplanes, self.PLANES[0], dimension=D)
        self.inplanes = self.PLANES[0]
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])
        self.attn2p2 = self.LAYER(self.inplanes, self.PLANES[1], dimension=D)
        self.inplanes = self.PLANES[1]
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])
        self.attn3p4 = self.LAYER(self.inplanes, self.PLANES[2])
        self.inplanes = self.PLANES[2]
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])
        self.attn4p8 = self.LAYER(self.inplanes, self.PLANES[3], dimension=D)
        self.inplanes = self.PLANES[3]
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        self.inplanes += self.PLANES[3] * self.BLOCK.expansion
        self.attn5p8 = self.LAYER(self.inplanes, self.PLANES[4], dimension=D)
        self.inplanes = self.PLANES[4]
        self.bn5 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])

        self.inplanes += self.PLANES[2] * self.BLOCK.expansion
        self.attn6p4 = self.LAYER(self.inplanes, self.PLANES[5], dimension=D)
        self.inplanes = self.PLANES[5]
        self.bn6 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])

        self.inplanes += self.PLANES[1] * self.BLOCK.expansion
        self.attn7p2 = self.LAYER(self.inplanes, self.PLANES[6], dimension=D)
        self.inplanes = self.PLANES[6]
        self.bn7 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])

        self.inplanes += self.PLANES[0]
        self.attn8p1 = self.LAYER(self.inplanes, self.PLANES[7], dimension=D)
        self.inplanes = self.PLANES[7]
        self.bn8 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=D
        )
    
        self.pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D)
        self.pooltr = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.TensorField):
        out = self.mlp(x)
        out, emb = self.voxelize(out)
        out_p1 = self.relu(self.bn1(self.attn1p1(out)))

        out = self.pool(out_p1)
        out = self.block1(out)
        out_p2 = self.relu(self.bn2(self.attn2p2(out)))

        out = self.pool(out_p2)
        out = self.block2(out)
        out_p4 = self.relu(self.bn3(self.attn3p4(out)))

        out = self.pool(out_p4)
        out = self.block3(out)
        out_p8 = self.relu(self.bn4(self.attn4p8(out)))

        out = self.pool(out_p8)
        out = self.block4(out)

        out = self.pooltr(out)
        out = ME.cat(out, out_p8)
        out = self.relu(self.bn5(self.attn5p8(out)))
        out = self.block5(out)

        out = self.pooltr(out)
        out = ME.cat(out, out_p4)
        out = self.relu(self.bn6(self.attn6p4(out)))
        out = self.block6(out)

        out = self.pooltr(out)
        out = ME.cat(out, out_p2)
        out = self.relu(self.bn7(self.attn7p2(out)))
        out = self.block7(out)

        out = self.pooltr(out)
        out = ME.cat(out, out_p1)
        out = self.relu(self.bn8(self.attn8p1(out)))
        out = self.block8(out)
        return self.devoxelize(out, x, emb)


@gin.configurable
class TransformerS(TransformerBase):
    BLOCK = MultiHeadSelfAttentionBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (64, 128, 256, 512, 256, 128, 64, 32)

    def voxelize(self, x: ME.TensorField):
        return x.sparse(), None
    
    def devoxelize(self, out: ME.SparseTensor, x: ME.TensorField, emb: torch.Tensor):
        return self.final(out).slice(x).F