#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "vector_attention_kernel.h"

void vector_attention_forward(
    int n, int c, int k, int s, AT in_out_tensor, AT in_cnt_tensor, AT query_tensor, AT value_tensor, AT kernel_tensor, AT out_F_tensor
    )
{
    const int* in_out = in_out_tensor.data_ptr<int>();
    const int* in_cnt = in_cnt_tensor.data_ptr<int>();
    const float* query = query_tensor.data_ptr<float>();
    const float* value = value_tensor.data_ptr<float>();
    const float* kernel = kernel_tensor.data_ptr<float>();
    float* out_F = out_F_tensor.data_ptr<float>();

    vector_attention_forward_launcher(
        n, c, k, s, in_out, in_cnt, query, value, kernel, out_F
    );
}

void vector_attention_backward(
    int n, int c, int k, int s, AT in_out_tensor, AT in_cnt_tensor, AT query_tensor, AT value_tensor, AT kernel_tensor,
    AT grad_output_tensor, AT grad_query_tensor, AT grad_value_tensor, AT grad_kernel_tensor
    )
{
    const int* in_out = in_out_tensor.data_ptr<int>();
    const int* in_cnt = in_cnt_tensor.data_ptr<int>();
    const float* query = query_tensor.data_ptr<float>();
    const float* value = value_tensor.data_ptr<float>();
    const float* kernel = kernel_tensor.data_ptr<float>();
    const float* grad_output = grad_output_tensor.data_ptr<float>();
    float* grad_query = grad_query_tensor.data_ptr<float>();
    float* grad_value = grad_value_tensor.data_ptr<float>();
    float* grad_kernel = grad_kernel_tensor.data_ptr<float>();
    
    vector_attention_backward_launcher(
        n, c, k, s, in_out, in_cnt, query, value, kernel, grad_output, grad_query, grad_value, grad_kernel
    );
}