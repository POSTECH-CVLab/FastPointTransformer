#pragma once
#ifndef _dot_product_intra_KERNEL
#define _dot_product_intra_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#define AT at::Tensor

void dot_product_intra_forward(
    int m, int h, int c, AT pos_tensor, AT out_F_tensor, AT kq_indices_tensor
    );
void dot_product_intra_backward(
    int m, int h, int c, AT pos_tensor, AT kq_indices_tensor,
    AT grad_pos_tensor, AT grad_out_F_tensor
    );

#ifdef __cplusplus
extern "C" {
#endif

void dot_product_intra_forward_launcher(
    int m, int h, int c, const float* pos, float* out_F, const int* kq_indices
    );
void dot_product_intra_backward_launcher(
    int m, int h, int c, const float* pos, const int* kq_indices,
    float* grad_pos, const float* grad_out_F
    );

#ifdef __cplusplus
}
#endif
#endif
