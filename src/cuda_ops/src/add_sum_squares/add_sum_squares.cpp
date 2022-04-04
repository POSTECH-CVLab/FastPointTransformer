#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "add_sum_squares_kernel.h"

void add_sum_squares_forward(
    int m, int h, int kkk, AT ss_key_tensor, AT ss_pos_tensor, AT out_F_tensor, AT k_map_tensor
    )
{
    const float* ss_key = ss_key_tensor.data_ptr<float>();
    const float* ss_pos = ss_pos_tensor.data_ptr<float>();
    float* out_F = out_F_tensor.data_ptr<float>();
    const int* k_map = k_map_tensor.data_ptr<int>();

    add_sum_squares_forward_launcher(
        m, h, kkk, ss_key, ss_pos, out_F, k_map
    );
}

void add_sum_squares_backward(
    int m, int h, int kkk, AT ss_key_tensor, AT ss_pos_tensor, AT k_map_tensor,
    AT grad_ss_key_tensor, AT grad_ss_pos_tensor, AT grad_out_F_tensor
    )
{
    const float* ss_key = ss_key_tensor.data_ptr<float>();
    const float* ss_pos = ss_pos_tensor.data_ptr<float>();
    const int* k_map = k_map_tensor.data_ptr<int>();

    float* grad_ss_key = grad_ss_key_tensor.data_ptr<float>();
    float* grad_ss_pos = grad_ss_pos_tensor.data_ptr<float>();
    const float* grad_out_F = grad_out_F_tensor.data_ptr<float>();
    
    add_sum_squares_backward_launcher(
        m, h, kkk, ss_key, ss_pos, k_map,
        grad_ss_key, grad_ss_pos, grad_out_F
    );
}