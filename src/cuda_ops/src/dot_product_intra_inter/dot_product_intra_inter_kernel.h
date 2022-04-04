#pragma once
#ifndef _dot_product_intra_inter_KERNEL
#define _dot_product_intra_inter_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#define AT at::Tensor

void dot_product_intra_inter_forward(
    int m, int h, int kkk, int c, AT query_tensor, AT intra_pos_tensor, AT inter_pos_tensor, AT out_F_tensor, AT kq_map_tensor
    );
void dot_product_intra_inter_backward(
    int m, int h, int kkk, int c, AT query_tensor, AT intra_pos_tensor, AT inter_pos_tensor, AT kq_map_tensor,
    AT grad_query_tensor, AT grad_intra_pos_tensor, AT grad_inter_pos_tensor, AT grad_out_F_tensor
    );

#ifdef __cplusplus
extern "C" {
#endif

void dot_product_intra_inter_forward_launcher(
    int m, int h, int kkk, int c, const float* query, const float* intra_pos, const float* inter_pos, float* out_F, const int* kq_map
    );
void dot_product_intra_inter_backward_launcher(
    int m, int h, int kkk, int c, const float* query, const float* intra_pos, const float* inter_pos, const int* kq_map,
    float* grad_query, float* grad_intra_pos, float* grad_inter_pos, const float* grad_out_F
    );

#ifdef __cplusplus
}
#endif
#endif
