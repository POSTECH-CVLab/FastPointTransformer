#include "../cuda_utils.h"
#include "add_sum_squares_kernel.h"


__global__ void add_sum_squares_forward_kernel(
    int m, int h, int kkk, const float* ss_key, const float* ss_pos, float* out_F, const int* k_map
)
{
    // m: # of total mappings
    // h: # of attention heads
    // kkk: # of keys (kernel volume)
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * h) return;

    int map_idx = index / h;
    int head_idx = index % h;

    int key_idx_ = k_map[map_idx] / kkk;
    int kernel_idx = k_map[map_idx] % kkk;

    int key_idx = key_idx_ * h + head_idx;
    int pos_idx = kernel_idx * h + head_idx;

    out_F[index] += ss_key[key_idx] + ss_pos[pos_idx];
}

__global__ void add_sum_squares_backward_kernel(
    int m, int h, int kkk, const float* ss_key, const float* ss_pos, const int* k_map,
    float* grad_ss_key, float* grad_ss_pos, const float* grad_out_F
)
{
    // m: # of total mappings
    // h: # of attention heads
    // kkk: # of keys (kernel volume)
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * h) return;

    int map_idx = index / h;
    int head_idx = index % h;

    int key_idx_ = k_map[map_idx] / kkk;
    int kernel_idx = k_map[map_idx] % kkk;

    int key_idx = key_idx_ * h + head_idx;
    int pos_idx = kernel_idx * h + head_idx;

    atomicAdd(grad_ss_key + key_idx, grad_out_F[index]);
    atomicAdd(grad_ss_pos + pos_idx, grad_out_F[index]);
}

void add_sum_squares_forward_launcher(
    int m, int h, int kkk, const float* ss_key, const float* ss_pos, float* out_F, const int* k_map
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * h, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    add_sum_squares_forward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, kkk, ss_key, ss_pos, out_F, k_map
    );
}

void add_sum_squares_backward_launcher(
    int m, int h, int kkk, const float* ss_key, const float* ss_pos, const int* k_map,
    float* grad_ss_key, float* grad_ss_pos, const float* grad_out_F
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * h, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    add_sum_squares_backward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, kkk, ss_key, ss_pos, k_map,
        grad_ss_key, grad_ss_pos, grad_out_F
    );
}
