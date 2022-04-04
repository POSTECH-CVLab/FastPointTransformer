import gin
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock


@gin.configurable
class ResNetBase(nn.Module):
    QMODE = ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
    LAYER = ME.MinkowskiConvolution
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels=None, D=3):
        nn.Module.__init__(self)
        assert self.BLOCK is not None
        self.D = D

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            self.LAYER(in_channels, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=D),
        )

        self.layer1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2)
        self.layer2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2)
        self.conv5 = nn.Sequential(
            ME.MinkowskiDropout(),
            ME.MinkowskiConvolution(self.inplanes,
                                    self.inplanes,
                                    kernel_size=3,
                                    stride=3,
                                    dimension=D),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiGELU(),
        )
        self.glob_pool = ME.MinkowskiGlobalMaxPooling()
        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ME.MinkowskiLinear):
                nn.init.xavier_normal_(m.linear.weight)
                if m.linear.bias is not None:
                    nn.init.constant_(m.linear.bias, 0)
            elif isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.LAYER(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D))
        return nn.Sequential(*layers)

    def voxelize(self, x: ME.TensorField):
        return x.sparse()

    def forward(self, x: ME.TensorField):
        x = self.voxelize(x)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.glob_pool(x)
        return self.final(x).F


@gin.configurable
class ResNet34(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)