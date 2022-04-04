import torch
import torch.nn as nn
from MinkowskiEngine.MinkowskiKernelGenerator import KernelGenerator


class LocalSelfAttentionBase(nn.Module):
    def __init__(self, kernel_size, stride, dilation, dimension):
        super(LocalSelfAttentionBase, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dimension = dimension
        
        self.kernel_generator = KernelGenerator(kernel_size=kernel_size,
                                                stride=stride,
                                                dilation=dilation,
                                                dimension=dimension)
        self.kernel_volume = self.kernel_generator.kernel_volume

    def get_kernel_map_and_out_key(self, stensor):
        cm = stensor.coordinate_manager
        in_key = stensor.coordinate_key
        out_key = cm.stride(in_key, self.kernel_generator.kernel_stride)
        region_type, region_offset, _ = self.kernel_generator.get_kernel(
            stensor.tensor_stride, False)
        kernel_map = cm.kernel_map(in_key,
                                   out_key,
                                   self.kernel_generator.kernel_stride,
                                   self.kernel_generator.kernel_size,
                                   self.kernel_generator.kernel_dilation,
                                   region_type=region_type,
                                   region_offset=region_offset)
        return kernel_map, out_key

    def key_query_map_from_kernel_map(self, kernel_map):
        kq_map = []
        for kernel_idx, in_out in kernel_map.items():
            in_out[0] = in_out[0] * self.kernel_volume + kernel_idx
            kq_map.append(in_out)
        kq_map = torch.cat(kq_map, -1)
        return kq_map
    
    def key_query_indices_from_kernel_map(self, kernel_map):
        kq_indices = []
        for _, in_out in kernel_map.items():
            kq_indices.append(in_out)
            kq_indices = torch.cat(kq_indices, -1)
        return kq_indices

    def key_query_indices_from_key_query_map(self, kq_map):
        kq_indices = kq_map.clone()
        kq_indices[0] = kq_indices[0] // self.kernel_volume
        return kq_indices