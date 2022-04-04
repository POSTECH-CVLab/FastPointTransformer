#include "../cuda_utils.h"
#include "dot_product_intra_kernel.h"


__global__ void dot_product_intra_forward_kernel(
    int m, int h, int c, const float* pos, float* out_F, const int* kq_indices
)
{
    // m: # of total mappings
    // h: # of attention heads
    // c: # of attention channels
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * h) return;

    int map_idx = index / h;
    int head_idx = index % h;

    int from_idx_ = kq_indices[map_idx]; // key index
    int to_idx_ = kq_indices[m + map_idx]; // query index

    for(int i = 0; i < c; i++){

        int from_idx = from_idx_ * h * c + head_idx * c + i;
        int to_idx = to_idx_ * h * c + head_idx * c + i;

        out_F[index] += pos[from_idx] * pos[to_idx];
    }
}

__global__ void dot_product_intra_backward_kernel(
    int m, int h, int c, const float* pos, const int* kq_indices,
    float* grad_pos, const float* grad_out_F
)
{
    // m: # of total mappings
    // h: # of attention heads
    // c: # of attention channels
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * c) return;

    int map_idx = index / c;
    int i = index % c;

    int from_idx_ = kq_indices[map_idx]; // key index
    int to_idx_ = kq_indices[m + map_idx]; // query index

    for(int head_idx = 0; head_idx < h; head_idx++){

        int out_F_idx = map_idx * h + head_idx;
        int from_idx = from_idx_ * h * c + head_idx * c + i;
        int to_idx = to_idx_ * h * c + head_idx * c + i;

        atomicAdd(grad_pos + from_idx, grad_out_F[out_F_idx] * pos[to_idx]);
        atomicAdd(grad_pos + to_idx, grad_out_F[out_F_idx] * pos[from_idx]);
    }
}

void dot_product_intra_forward_launcher(
    int m, int h, int c, const float* pos, float* out_F, const int* kq_indices
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * h, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    dot_product_intra_forward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, c, pos, out_F, kq_indices
    );
}

void dot_product_intra_backward_launcher(
    int m, int h, int c, const float* pos, const int* kq_indices,
    float* grad_pos, const float* grad_out_F
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    dot_product_intra_backward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, c, pos, kq_indices,
        grad_pos, grad_out_F
    );
}
