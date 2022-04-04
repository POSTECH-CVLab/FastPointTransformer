#include "../cuda_utils.h"
#include "dot_product_key_kernel.h"


__global__ void dot_product_key_forward_kernel(
    int m, int h, int kkk, int c, const float* key, const float* pos, float* out_F, const int* k_map
)
{
    // m: # of total mappings
    // h: # of attention heads
    // kkk: # of keys (kernel volume)
    // c: # of attention channels
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * h) return;

    int map_idx = index / h;
    int head_idx = index % h;

    int key_idx_ = k_map[map_idx] / kkk;
    int kernel_idx = k_map[map_idx] % kkk;

    for(int i = 0; i < c; i++){

        int key_idx = key_idx_ * h * c + head_idx * c + i;
        int pos_idx = kernel_idx * h * c + head_idx * c + i;

        out_F[index] += key[key_idx] * pos[pos_idx];
    }
}

__global__ void dot_product_key_backward_kernel(
    int m, int h, int kkk, int c, const float* key, const float* pos, const int* k_map,
    float* grad_key, float* grad_pos, const float* grad_out_F
)
{
    // m: # of total mappings
    // h: # of attention heads
    // kkk: # of keys (kernel volume)
    // c: # of attention channels
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * c) return;

    int map_idx = index / c;
    int i = index % c;

    int key_idx_ = k_map[map_idx] / kkk;
    int kernel_idx = k_map[map_idx] % kkk;

    for(int head_idx = 0; head_idx < h; head_idx++){

        int out_F_idx = map_idx * h + head_idx;
        int key_idx = key_idx_ * h * c + head_idx * c + i;
        int pos_idx = kernel_idx * h * c + head_idx * c + i;

        atomicAdd(grad_key + key_idx, grad_out_F[out_F_idx] * pos[pos_idx]);
        atomicAdd(grad_pos + pos_idx, grad_out_F[out_F_idx] * key[key_idx]);
    }
}

void dot_product_key_forward_launcher(
    int m, int h, int kkk, int c, const float* key, const float* pos, float* out_F, const int* k_map
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * h, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    dot_product_key_forward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, kkk, c, key, pos, out_F, k_map
    );
}

void dot_product_key_backward_launcher(
    int m, int h, int kkk, int c, const float* key, const float* pos, const int* k_map,
    float* grad_key, float* grad_pos, const float* grad_out_F
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    dot_product_key_backward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, kkk, c, key, pos, k_map,
        grad_key, grad_pos, grad_out_F
    );
}
