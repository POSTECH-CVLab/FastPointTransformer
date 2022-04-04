#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "scalar_attention_kernel.h"

void scalar_attention_forward(
    int m, int h, int c, AT weight_tensor, AT value_tensor, AT out_F_tensor, AT kq_indices_tensor
    )
{
    const float* weight = weight_tensor.data_ptr<float>();
    const float* value = value_tensor.data_ptr<float>();
    float* out_F = out_F_tensor.data_ptr<float>();
    const int* kq_indices = kq_indices_tensor.data_ptr<int>();

    scalar_attention_forward_launcher(
        m, h, c, weight, value, out_F, kq_indices
    );
}

void scalar_attention_backward(
    int m, int h, int c, AT weight_tensor, AT value_tensor, AT kq_indices_tensor,
    AT grad_weight_tensor, AT grad_value_tensor, AT grad_out_F_tensor
    )
{
    const float* weight = weight_tensor.data_ptr<float>();
    const float* value = value_tensor.data_ptr<float>();
    const int* kq_indices = kq_indices_tensor.data_ptr<int>();

    float* grad_weight = grad_weight_tensor.data_ptr<float>();
    float* grad_value = grad_value_tensor.data_ptr<float>();
    const float* grad_out_F = grad_out_F_tensor.data_ptr<float>();
    
    scalar_attention_backward_launcher(
        m, h, c, weight, value, kq_indices,
        grad_weight, grad_value, grad_out_F
    );
}