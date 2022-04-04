#pragma once
#ifndef _add_sum_squares_KERNEL
#define _add_sum_squares_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#define AT at::Tensor

void add_sum_squares_forward(
    int m, int h, int kkk, AT ss_key_tensor, AT ss_pos_tensor, AT out_F_tensor, AT k_map_tensor
    );
void add_sum_squares_backward(
    int m, int h, int kkk, AT ss_key_tensor, AT ss_pos_tensor, AT k_map_tensor,
    AT grad_ss_key_tensor, AT grad_ss_pos_tensor, AT grad_out_F_tensor
    );

#ifdef __cplusplus
extern "C" {
#endif

void add_sum_squares_forward_launcher(
    int m, int h, int kkk, const float* ss_key, const float* ss_pos, float* out_F, const int* k_map
    );
void add_sum_squares_backward_launcher(
    int m, int h, int kkk, const float* ss_key, const float* ss_pos, const int* k_map,
    float* grad_ss_key, float* grad_ss_pos, const float* grad_out_F
    );

#ifdef __cplusplus
}
#endif
#endif
