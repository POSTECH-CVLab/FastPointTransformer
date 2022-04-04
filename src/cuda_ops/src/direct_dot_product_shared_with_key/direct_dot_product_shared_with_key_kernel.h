#pragma once
#ifndef _direct_dot_product_shared_with_key_KERNEL
#define _direct_dot_product_shared_with_key_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#define AT at::Tensor

void direct_dot_product_shared_with_key_forward(
    int m, int h, int c, AT query_tensor, AT key_tensor, AT pos_tensor, AT out_F_tensor, AT kq_indices_tensor
    );
void direct_dot_product_shared_with_key_backward(
    int m, int h, int c, AT query_tensor, AT key_tensor, AT pos_tensor, AT kq_indices_tensor,
    AT grad_query_tensor, AT grad_key_tensor, AT grad_pos_tensor, AT grad_out_F_tensor
    );

#ifdef __cplusplus
extern "C" {
#endif

void direct_dot_product_shared_with_key_forward_launcher(
    int m, int h, int c, const float* query, const float* key, const float* pos, float* out_F, const int* kq_indices
    );
void direct_dot_product_shared_with_key_backward_launcher(
    int m, int h, int c, const float* query, const float* key, const float* pos, const int* kq_indices,
    float* grad_query, float* grad_key, float* grad_pos, const float* grad_out_F
    );

#ifdef __cplusplus
}
#endif
#endif
