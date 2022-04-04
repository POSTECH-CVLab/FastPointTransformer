#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "dot_product_sample_with_key_kernel.h"

void dot_product_sample_with_key_forward(
    int m, int h, int c, AT query_tensor, AT key_tensor, AT pos_tensor, AT out_F_tensor, AT skq_indices_tensor
    )
{
    const float* query = query_tensor.data_ptr<float>();
    const float* key = key_tensor.data_ptr<float>();
    const float* pos = pos_tensor.data_ptr<float>();
    float* out_F = out_F_tensor.data_ptr<float>();
    const int* skq_indices = skq_indices_tensor.data_ptr<int>();

    dot_product_sample_with_key_forward_launcher(
        m, h, c, query, key, pos, out_F, skq_indices
    );
}

void dot_product_sample_with_key_backward(
    int m, int h, int c, AT query_tensor, AT key_tensor, AT pos_tensor, AT skq_indices_tensor,
    AT grad_query_tensor, AT grad_key_tensor, AT grad_pos_tensor, AT grad_out_F_tensor
    )
{
    const float* query = query_tensor.data_ptr<float>();
    const float* key = key_tensor.data_ptr<float>();
    const float* pos = pos_tensor.data_ptr<float>();
    const int* skq_indices = skq_indices_tensor.data_ptr<int>();

    float* grad_query = grad_query_tensor.data_ptr<float>();
    float* grad_key = grad_key_tensor.data_ptr<float>();
    float* grad_pos = grad_pos_tensor.data_ptr<float>();
    const float* grad_out_F = grad_out_F_tensor.data_ptr<float>();
    
    dot_product_sample_with_key_backward_launcher(
        m, h, c, query, key, pos, skq_indices,
        grad_query, grad_key, grad_pos, grad_out_F
    );
}