#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "dot_product_key_kernel.h"

void dot_product_key_forward(
    int m, int h, int kkk, int c, AT key_tensor, AT pos_tensor, AT out_F_tensor, AT k_map_tensor
    )
{
    const float* key = key_tensor.data_ptr<float>();
    const float* pos = pos_tensor.data_ptr<float>();
    float* out_F = out_F_tensor.data_ptr<float>();
    const int* k_map = k_map_tensor.data_ptr<int>();

    dot_product_key_forward_launcher(
        m, h, kkk, c, key, pos, out_F, k_map
    );
}

void dot_product_key_backward(
    int m, int h, int kkk, int c, AT key_tensor, AT pos_tensor, AT k_map_tensor,
    AT grad_key_tensor, AT grad_pos_tensor, AT grad_out_F_tensor
    )
{
    const float* key = key_tensor.data_ptr<float>();
    const float* pos = pos_tensor.data_ptr<float>();
    const int* k_map = k_map_tensor.data_ptr<int>();

    float* grad_key = grad_key_tensor.data_ptr<float>();
    float* grad_pos = grad_pos_tensor.data_ptr<float>();
    const float* grad_out_F = grad_out_F_tensor.data_ptr<float>();
    
    dot_product_key_backward_launcher(
        m, h, kkk, c, key, pos, k_map,
        grad_key, grad_pos, grad_out_F
    );
}