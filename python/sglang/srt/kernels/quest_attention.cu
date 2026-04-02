#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_HEAD_DIM 256

__global__ void quest_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    const int*   __restrict__ selected_pages,
    float* __restrict__ output,
    int num_selected,
    int page_size,
    int head_dim,
    int num_heads,
    int num_tokens
) {
    int bh = blockIdx.x;
    int b = bh / num_heads;
    int h = bh % num_heads;
    int tid = threadIdx.x;

    const float* Q_bh = Q + (b * num_heads + h) * head_dim;
    const int* pages_bh = selected_pages + (b * num_heads + h) * num_selected;
    float* out_bh = output + (b * num_heads + h) * head_dim;

    int kv_stride = num_heads * head_dim;

    __shared__ float Q_shared[MAX_HEAD_DIM];
    __shared__ float scores[BLOCK_SIZE];
    __shared__ float reduce_buf[BLOCK_SIZE];

    for (int i = tid; i < head_dim; i += BLOCK_SIZE) {
        Q_shared[i] = Q_bh[i];
    }
    __syncthreads();

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float global_max = -INFINITY;
    float global_sum = 0.0f;

    for (int p = 0; p < num_selected; p++) {
        int page_idx = pages_bh[p];
        int token_start = page_idx * page_size;
        int valid_tokens = min(page_size, num_tokens - token_start);

        float my_score = -INFINITY;
        if (tid < valid_tokens) {
            int token_idx = token_start + tid;
            const float* K_token = K_cache + token_idx * kv_stride + h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += Q_shared[d] * K_token[d];
            }
            my_score = dot;
        }

        // Max reduction
        reduce_buf[tid] = my_score;
        __syncthreads(); 
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
            }
            __syncthreads();
        }
        float page_max = reduce_buf[0];
        __syncthreads(); 

        // Sum reduction
        float my_exp = (my_score == -INFINITY) ? 0.0f : expf(my_score - page_max);
        reduce_buf[tid] = my_exp;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] += reduce_buf[tid + stride];
            }
            __syncthreads();
        }
        float page_sum = reduce_buf[0];
        // No sync needed here: reduce_buf isn't written again until next iteration,
        // which is separated by two __syncthreads__ below.

        // Online softmax merge
        float new_max = fmaxf(global_max, page_max);
        float old_scale = (global_max == -INFINITY) ? 0.0f : expf(global_max - new_max);
        float new_scale = expf(page_max - new_max);

        for (int i = 0; i < 4; i++) {
            acc[i] *= old_scale;
        }
        global_sum = global_sum * old_scale + page_sum * new_scale;
        global_max = new_max;

        // Store weights for V accumulation
        scores[tid] = my_exp * new_scale;
        __syncthreads();

        // Accumulate weighted V
        for (int t = 0; t < valid_tokens; t++) {
            float w = scores[t];
            if (w == 0.0f) continue;
            int token_idx = token_start + t;
            const float* V_token = V_cache + token_idx * kv_stride + h * head_dim;
            for (int i = 0; i < 4; i++) {
                int d = tid + i * BLOCK_SIZE;
                if (d < head_dim) {
                    acc[i] += w * V_token[d];
                }
            }
        }
        __syncthreads();
    }

    if (global_sum > 0.0f) {
        for (int i = 0; i < 4; i++) {
            int d = tid + i * BLOCK_SIZE;
            if (d < head_dim) {
                out_bh[d] = acc[i] / global_sum;
            }
        }
    }
}

torch::Tensor quest_attention_forward(
    torch::Tensor Q,
    torch::Tensor K_cache,
    torch::Tensor V_cache,
    torch::Tensor selected_pages,
    int page_size
) {
    int batch_size = Q.size(0);
    int num_heads = Q.size(1);
    int head_dim = Q.size(2);
    int num_selected = selected_pages.size(2);
    int num_tokens = K_cache.size(0);

    TORCH_CHECK(head_dim <= MAX_HEAD_DIM * 4, "head_dim too large");
    TORCH_CHECK(page_size <= BLOCK_SIZE, "page_size must be <= BLOCK_SIZE");

    auto output = torch::zeros({batch_size, num_heads, head_dim}, Q.options());

    quest_attention_kernel<<<batch_size * num_heads, BLOCK_SIZE>>>(
        Q.data_ptr<float>(),
        K_cache.data_ptr<float>(),
        V_cache.data_ptr<float>(),
        selected_pages.data_ptr<int>(),
        output.data_ptr<float>(),
        num_selected,
        page_size,
        head_dim,
        num_heads,
        num_tokens
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quest_attention_forward", &quest_attention_forward, "QUEST attention forward (CUDA)");
}