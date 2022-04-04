#include "../cuda_utils.h"
#include "scalar_attention_kernel.h"


__global__ void scalar_attention_forward_kernel(
    int m, int h, int c, const float* weight, const float* value, float* out_F, const int* kq_indices
)
{
    // m: # of total mappings
    // h: # of attention heads
    // c: # of attention channels
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * c) return;

    int map_idx = index / c;
    int i = index % c;

    int out_F_idx_ = kq_indices[m + map_idx]; // kq_indices[1][map_idx]
    int value_idx_ = kq_indices[map_idx]; // kq_indices[0][map_idx]

    for(int head_idx = 0; head_idx < h; head_idx++){

        int weight_idx = map_idx * h + head_idx;
        int out_F_idx = out_F_idx_ * h * c + head_idx * c + i;
        int value_idx = value_idx_ * h * c + head_idx * c + i;

        atomicAdd(out_F + out_F_idx, weight[weight_idx] * value[value_idx]);
    }
}

__global__ void scalar_attention_backward_kernel(
    int m, int h, int c, const float* weight, const float* value, const int* kq_indices,
    float* grad_weight, float* grad_value, const float* grad_out_F
)
{
    // m: # of total mappings
    // h: # of attention heads
    // c: # of attention channels
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * c) return;

    int map_idx = index / c;
    int i = index % c;

    int out_F_idx_ = kq_indices[m + map_idx]; // kq_indices[1][map_idx]
    int value_idx_ = kq_indices[map_idx]; // kq_indices[0][map_idx]

    for(int head_idx = 0; head_idx < h; head_idx++){

        int weight_idx = map_idx * h + head_idx;
        int out_F_idx = out_F_idx_ * h * c + head_idx * c + i;
        int value_idx = value_idx_ * h * c + head_idx * c + i;

        atomicAdd(grad_weight + weight_idx, grad_out_F[out_F_idx] * value[value_idx]);
        atomicAdd(grad_value + value_idx, grad_out_F[out_F_idx] * weight[weight_idx]);
    }
}

void scalar_attention_forward_launcher(
    int m, int h, int c, const float* weight, const float* value, float* out_F, const int* kq_indices
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    scalar_attention_forward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, c, weight, value, out_F, kq_indices
    );
}

void scalar_attention_backward_launcher(
    int m, int h, int c, const float* weight, const float* value, const int* kq_indices,
    float* grad_weight, float* grad_value, const float* grad_out_F
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    scalar_attention_backward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, c, weight, value, kq_indices,
        grad_weight, grad_value, grad_out_F
    );
}
