#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "dot_product_intra_kernel.h"

void dot_product_intra_forward(
    int m, int h, int c, AT pos_tensor, AT out_F_tensor, AT kq_indices_tensor
    )
{
    const float* pos = pos_tensor.data_ptr<float>();
    float* out_F = out_F_tensor.data_ptr<float>();
    const int* kq_indices = kq_indices_tensor.data_ptr<int>();

    dot_product_intra_forward_launcher(
        m, h, c, pos, out_F, kq_indices
    );
}

void dot_product_intra_backward(
    int m, int h, int c, AT pos_tensor, AT kq_indices_tensor,
    AT grad_pos_tensor, AT grad_out_F_tensor
    )
{
    const float* pos = pos_tensor.data_ptr<float>();
    const int* kq_indices = kq_indices_tensor.data_ptr<int>();

    float* grad_pos = grad_pos_tensor.data_ptr<float>();
    const float* grad_out_F = grad_out_F_tensor.data_ptr<float>();
    
    dot_product_intra_backward_launcher(
        m, h, c, pos, kq_indices,
        grad_pos, grad_out_F
    );
}