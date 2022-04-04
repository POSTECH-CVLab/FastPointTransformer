#include "../cuda_utils.h"
#include "dot_product_sample_shared_kernel.h"


__global__ void dot_product_sample_shared_forward_kernel(
    int m, int h, int c, const float* query, const float* pos, float* out_F, const int* sq_indices
)
{
    // m: # of total mappings
    // h: # of attention heads
    // c: # of attention channels
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * h) return;

    int map_idx = index / h;
    int head_idx = index % h;

    int sample_idx = sq_indices[map_idx]; // sq_indices[0][map_idx]
    int query_idx_ = sq_indices[m + map_idx]; // sq_indices[1][map_idx]

    for(int i = 0; i < c; i++){

        int query_idx = query_idx_ * h * c + head_idx * c + i;
        int pos_idx = sample_idx * c + i;

        out_F[index] += query[query_idx] * pos[pos_idx];
    }
}

__global__ void dot_product_sample_shared_backward_kernel(
    int m, int h, int c, const float* query, const float* pos, const int* sq_indices,
    float* grad_query, float* grad_pos, const float* grad_out_F
)
{
    // m: # of total mappings
    // h: # of attention heads
    // c: # of attention channels
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * c) return;

    int map_idx = index / c;
    int i = index % c;

    int sample_idx = sq_indices[map_idx]; // sq_indices[0][map_idx]
    int query_idx_ = sq_indices[m + map_idx]; // sq_indices[1][map_idx]

    for(int head_idx = 0; head_idx < h; head_idx++){

        int out_F_idx = map_idx * h + head_idx;
        int query_idx = query_idx_ * h * c + head_idx * c + i;
        int pos_idx = sample_idx * c + i;

        atomicAdd(grad_query + query_idx, grad_out_F[out_F_idx] * pos[pos_idx]);
        atomicAdd(grad_pos + pos_idx, grad_out_F[out_F_idx] * query[query_idx]);
    }
}

void dot_product_sample_shared_forward_launcher(
    int m, int h, int c, const float* query, const float* pos, float* out_F, const int* sq_indices
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * h, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    dot_product_sample_shared_forward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, c, query, pos, out_F, sq_indices
    );
}

void dot_product_sample_shared_backward_launcher(
    int m, int h, int c, const float* query, const float* pos, const int* sq_indices,
    float* grad_query, float* grad_pos, const float* grad_out_F
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    dot_product_sample_shared_backward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, c, query, pos, sq_indices,
        grad_query, grad_pos, grad_out_F
    );
}
