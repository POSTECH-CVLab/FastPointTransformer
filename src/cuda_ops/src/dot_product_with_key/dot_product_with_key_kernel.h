#pragma once
#ifndef _dot_product_with_key_KERNEL
#define _dot_product_with_key_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#define AT at::Tensor

void dot_product_with_key_forward(
    int m, int h, int kkk, int c, AT query_tensor, AT key_tensor, AT pos_tensor, AT out_F_tensor, AT kq_map_tensor
    );
void dot_product_with_key_backward(
    int m, int h, int kkk, int c, AT query_tensor, AT key_tensor, AT pos_tensor, AT kq_map_tensor,
    AT grad_query_tensor, AT grad_key_tensor, AT grad_pos_tensor, AT grad_out_F_tensor
    );

#ifdef __cplusplus
extern "C" {
#endif

void dot_product_with_key_forward_launcher(
    int m, int h, int kkk, int c, const float* query, const float* key, const float* pos, float* out_F, const int* kq_map
    );
void dot_product_with_key_backward_launcher(
    int m, int h, int kkk, int c, const float* query, const float* key, const float* pos, const int* kq_map,
    float* grad_query, float* grad_key, float* grad_pos, const float* grad_out_F
    );

#ifdef __cplusplus
}
#endif
#endif
