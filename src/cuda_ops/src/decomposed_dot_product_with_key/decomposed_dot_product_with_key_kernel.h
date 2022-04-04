#pragma once
#ifndef _decomposed_dot_product_with_key_KERNEL
#define _decomposed_dot_product_with_key_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#define AT at::Tensor

void decomposed_dot_product_with_key_forward(
    int m, int h, int kkk, int c, AT query_tensor, AT key_tensor, AT pos_intra_tensor, AT pos_inter_tensor, AT out_F_tensor, AT kq_map_tensor
    );
void decomposed_dot_product_with_key_backward(
    int m, int h, int kkk, int c, AT query_tensor, AT key_tensor, AT pos_intra_tensor, AT pos_inter_tensor, AT kq_map_tensor,
    AT grad_query_tensor, AT grad_key_tensor, AT grad_pos_intra_tensor, AT grad_pos_inter_tensor, AT grad_out_F_tensor
    );

#ifdef __cplusplus
extern "C" {
#endif

void decomposed_dot_product_with_key_forward_launcher(
    int m, int h, int kkk, int c, const float* query, const float* key, const float* pos_intra, const float* pos_inter, float* out_F, const int* kq_map
    );
void decomposed_dot_product_with_key_backward_launcher(
    int m, int h, int kkk, int c, const float* query, const float* key, const float* pos_intra, const float* pos_inter, const int* kq_map,
    float* grad_query, float* grad_key, float* grad_pos_intra, float* grad_pos_inter, const float* grad_out_F
    );

#ifdef __cplusplus
}
#endif
#endif
