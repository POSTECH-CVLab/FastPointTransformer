#include "../cuda_utils.h"
#include "vector_attention_kernel.h"


__global__ void vector_attention_forward_kernel(
    int n, int c, int k, int s, const int* in_out, const int* in_cnt, const float* query, const float* value, const float* kernel, float* out_F
)
{
    // n: # of query coordinates
    // c: # of channels
    // k: kernel volume
    // s: # of attention channels
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * c) return;

    int out_idx = index / c;
    int channel = index % c;
    int attn_channel = index % s;

    int from = out_idx == 0 ? 0 : in_cnt[out_idx - 1];
    int to = in_cnt[out_idx];

    for(int i = from; i < to; i++){
        
        int query_idx = out_idx * s + attn_channel;
        int kernel_idx = (in_out[i] % k) * s + attn_channel;
        int value_idx = (in_out[i] / k) * c + channel;

        out_F[index] += (query[query_idx] - kernel[kernel_idx]) * value[value_idx];
    }
}

__global__ void vector_attention_backward_kernel(
    int n, int c, int k, int s, const int* in_out, const int* in_cnt, const float* query, const float* value, const float* kernel, 
    const float* grad_output, float* grad_query, float* grad_value, float* grad_kernel
)
{
    // n: # of query coordinates
    // c: # of channels
    // k: kernel volume
    // s: # of attention channels
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * c) return;

    int out_idx = index / c;
    int channel = index % c;
    int attn_channel = index % s;

    int from = out_idx == 0 ? 0 : in_cnt[out_idx - 1]; 
    int to = in_cnt[out_idx];

    for (int i = from; i < to; i++){

        int query_idx = out_idx * s + attn_channel;
        int kernel_idx = (in_out[i] % k) * s + attn_channel;
        int value_idx = (in_out[i] / k) * c + channel;
        
        atomicAdd(grad_query + query_idx, grad_output[index] * value[value_idx]);
        atomicAdd(grad_kernel + kernel_idx, -grad_output[index] * value[value_idx]);
        atomicAdd(grad_value + value_idx, grad_output[index] * (query[query_idx] - kernel[kernel_idx]));
    }
}

void vector_attention_forward_launcher(
    int n, int c, int k, int s, const int* in_out, const int* in_cnt, const float* query, const float* value, const float* kernel, float* out_F
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(n * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    vector_attention_forward_kernel<<<blocks, threads, 0, stream>>>(
        n, c, k, s, in_out, in_cnt, query, value, kernel, out_F
    );
}

void vector_attention_backward_launcher(
    int n, int c, int k, int s, const int* in_out, const int* in_cnt, const float* query, const float* value, const float* kernel, 
    const float* grad_output, float* grad_query, float* grad_value, float* grad_kernel
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(DIVUP(n * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    vector_attention_backward_kernel<<<blocks, threads, 0, stream>>>(
        n, c, k, s, in_out, in_cnt, query, value, kernel, grad_output, grad_query, grad_value, grad_kernel
    );
}
