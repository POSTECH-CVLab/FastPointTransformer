#pragma once
#ifndef _scalar_attention_KERNEL
#define _scalar_attention_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#define AT at::Tensor

void scalar_attention_forward(
    int m, int h, int c, AT weight_tensor, AT value_tensor, AT out_F_tensor, AT kq_indices_tensor
    );
void scalar_attention_backward(
    int m, int h, int c, AT weight_tensor, AT value_tensor, AT kq_indices_tensor,
    AT grad_weight_tensor, AT grad_value_tensor, AT grad_out_F_tensor
    );

#ifdef __cplusplus
extern "C" {
#endif

void scalar_attention_forward_launcher(
    int m, int h, int c, const float* weight, const float* value, float* out_F, const int* kq_indices
    );
void scalar_attention_backward_launcher(
    int m, int h, int c, const float* weight, const float* value, const int* kq_indices,
    float* grad_weight, float* grad_value, const float* grad_out_F
    );

#ifdef __cplusplus
}
#endif
#endif
