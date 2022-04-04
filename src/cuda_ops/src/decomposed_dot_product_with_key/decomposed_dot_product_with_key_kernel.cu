#include "../cuda_utils.h"
#include "decomposed_dot_product_with_key_kernel.h"


__global__ void decomposed_dot_product_with_key_forward_kernel(
    int m, int h, int kkk, int c, const float* query, const float* key, const float* pos_intra, const float* pos_inter, float* out_F, const int* kq_map
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

    int query_idx_ = kq_map[m + map_idx]; // kq_map[1][map_idx]
    int key_idx_ = kq_map[map_idx] / kkk; // kq_map[0][map_idx] / kernel_volume
    int inter_idx_ = kq_map[map_idx] % kkk;

    for(int i = 0; i < c; i++){

        int query_idx = query_idx_ * h * c + head_idx * c + i;
        int key_idx = key_idx_ * h * c + head_idx * c + i;
        int pos_inter_idx = inter_idx_ * h * c + head_idx * c + i;

        out_F[index] += query[query_idx] * (key[key_idx] + pos_intra[query_idx] - pos_intra[key_idx] + pos_inter[pos_inter_idx]);
    }
}

__global__ void decomposed_dot_product_with_key_backward_kernel(
    int m, int h, int kkk, int c, const float* query, const float* key, const float* pos_intra, const float* pos_inter, const int* kq_map,
    float* grad_query, float* grad_key, float* grad_pos_intra, float* grad_pos_inter, const float* grad_out_F
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

    int query_idx_ = kq_map[m + map_idx]; // kq_map[1][map_idx]
    int key_idx_ = kq_map[map_idx] / kkk; // kq_map[0][map_idx] / kernel_volume
    int inter_idx_ = kq_map[map_idx] % kkk;

    for(int head_idx = 0; head_idx < h; head_idx++){

        int out_F_idx = map_idx * h + head_idx;
        int query_idx = query_idx_ * h * c + head_idx * c + i;
        int key_idx = key_idx_ * h * c + head_idx * c + i;
        int pos_inter_idx = inter_idx_ * h * c + head_idx * c + i;

        atomicAdd(grad_query + query_idx, grad_out_F[out_F_idx] * (key[key_idx] + pos_intra[query_idx] - pos_intra[key_idx] + pos_inter[pos_inter_idx]));
        atomicAdd(grad_key + key_idx, grad_out_F[out_F_idx] * query[query_idx]);
        atomicAdd(grad_pos_intra + query_idx, grad_out_F[out_F_idx] * query[query_idx]);
        atomicAdd(grad_pos_intra + key_idx, grad_out_F[out_F_idx] * query[query_idx]);
        atomicAdd(grad_pos_inter + pos_inter_idx, grad_out_F[out_F_idx] * query[query_idx]);
    }
}

void decomposed_dot_product_with_key_forward_launcher(
    int m, int h, int kkk, int c, const float* query, const float* key, const float* pos_intra, const float* pos_inter, float* out_F, const int* kq_map
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * h, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    decomposed_dot_product_with_key_forward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, kkk, c, query, key, pos_intra, pos_inter, out_F, kq_map
    );
}

void decomposed_dot_product_with_key_backward_launcher(
    int m, int h, int kkk, int c, const float* query, const float* key, const float* pos_intra, const float* pos_inter, const int* kq_map,
    float* grad_query, float* grad_key, float* grad_pos_intra, float* grad_pos_inter, const float* grad_out_F
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(m * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    decomposed_dot_product_with_key_backward_kernel<<<blocks, threads, 0, stream>>>(
        m, h, kkk, c, query, key, pos_intra, pos_inter, kq_map,
        grad_query, grad_key, grad_pos_intra, grad_pos_inter, grad_out_F
    );
}
