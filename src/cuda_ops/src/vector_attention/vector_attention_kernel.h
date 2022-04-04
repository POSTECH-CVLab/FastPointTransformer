#pragma once
#ifndef _vector_attention_KERNEL
#define _vector_attention_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#define AT at::Tensor

void vector_attention_forward(
    int n, int c, int k, int s, AT in_out_tensor, AT in_cnt_tensor, AT query_tensor, AT value_tensor, AT kernel_tensor, AT out_F_tensor
    );
void vector_attention_backward(
    int n, int c, int k, int s, AT in_out_tensor, AT in_cnt_tensor, AT query_tensor, AT value_tensor, AT kernel_tensor,
    AT grad_output_tensor, AT grad_query_tensor, AT grad_value_tensor, AT grad_kernel_tensor
    );

#ifdef __cplusplus
extern "C" {
#endif

void vector_attention_forward_launcher(
    int n, int c, int k, int s, const int* in_out_tensor, const int* in_cnt_tensor, const float* query, const float* value, const float* kernel, float* out_F
    );
void vector_attention_backward_launcher(
    int n, int c, int k, int s, const int* in_out_tensor, const int* in_cnt_tensor, const float* query, const float* value, const float* kernel, 
    const float* grad_output, float* grad_query, float* grad_value, float* grad_kernel
    );

#ifdef __cplusplus
}
#endif
#endif
