#include "../cuda_utils.h"
#include "direct_dot_product_with_key_kernel.h"


__global__ void direct_dot_product_with_key_forward_kernel(
    int m, int h, int c, const float* query, const float* key, const float* pos, float* out_F, const int* kq_indices
)
{
    // m: # of total mappings
    // h: # of attention heads
    // c: # of attention channels
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * h) return;

    int map_idx = index / h;
    int head_idx = index % h;

    int key_idx_ = kq_indices[map_idx]; // kq_indices[0][map_idx]
    int query_idx_ = kq_indices[m + map_idx]; // kq_indices[1][map_idx]

    for(int i = 0; i < c; i++){

        int key_idx = key_idx_ * h * c + head_idx * c + i;
        int query_idx = query_idx_ * h * c + head_idx * c + i;
        int pos_idx = map_idx * h * c + head_idx * c + i;

        out_F[index] += query[query_idx] * (key[key_idx] + pos[pos_idx]);
    }
}

__global__ void direct_dot_product_with_key_backward_kernel(
    int m, int h, int c, const float* query, const float* key, const float* pos, const int* kq_indices,
    float* grad_query, float* grad_key, float* grad_pos, const float* grad_out_F
)
{
    // m: # of total mappings
    // h: # of attention heads
    // c: # of attention channels
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * c) return;

    int map_idx = index / c;
    int i = index % c;

    int key_idx_ = kq_indices[map_idx]; // kq_indices[0][map_idx]
    int query_idx_ = kq_indices[m + map_idx]; // kq_indices[1][map_idx]

    for(int head_idx = 0; head_idx < h; head_idx++){

        int out_F_idx = map_idx * h + head_idx;
        int key_idx = key_idx_ * h * c + head_idx * c + i;
        int query_idx = query_idx_ * h * c + head_idx * c + i;
        int pos_idx = map_idx * h * c + head_idx * c + i;

        atomicAdd(grad_query + query_idx, grad_out_F[out_F_idx] * (key[key_idx] + pos[pos_idx]));
        atomicAdd(grad_key + key_idx, grad_out_F[out_F_idx] * query[query_idx]);
        atomicAdd(grad_pos + pos_idx, grad_out_F[out_F_idx] * query[query_idx]);
    }
}

void direct_dot_product_with_key_forward_launcher(
    int m, int h, int c, const float* query, const float* key, const float* pos, float* out_F, const int* kq_indices
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * h, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    direct_dot_product_with_key_forward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, c, query, key, pos, out_F, kq_indices
    );
}

void direct_dot_product_with_key_backward_launcher(
    int m, int h, int c, const float* query, const float* key, const float* pos, const int* kq_indices,
    float* grad_query, float* grad_key, float* grad_pos, const float* grad_out_F
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    direct_dot_product_with_key_backward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, c, query, key, pos, kq_indices,
        grad_query, grad_key, grad_pos, grad_out_F
    );
}
