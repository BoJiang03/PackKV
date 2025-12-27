#include "fused_kernels.h"
#include <type_traits.h>
#include <cstdio>

// Warp-level reduction using shuffle instructions
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T, int BLOCK_SIZE, int M_STATIC, int K_STATIC>
__global__
void fused_kq_kernel(
    T* __restrict__ our_kq,
    const T* __restrict__ q,
    const T* __restrict__ k_quant_zero,
    const T* __restrict__ k_quant_scale,
    const T* __restrict__ t,
    const int N
) {
    using Traits = TypeTraits<T>;
    using scalar_t = typename Traits::scalar_t;
    using vector_t = typename Traits::vector_t;
    
    const int m = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float s_mem[];

    // --- 1. Compute s = q[m, :].sum() using parallel reduction ---
    const scalar_t* q_row = q + m * K_STATIC;
    float thread_sum = 0.0f;

    // K_STATIC is 128, BLOCK_SIZE is 256.
    // Let each thread process one element. Only the first K_STATIC threads will work.
    // This is better than the original version where only K_STATIC/2 threads worked.
    if (tid < K_STATIC) {
        thread_sum = Traits::to_float(q_row[tid]);
    }
    s_mem[tid] = thread_sum;
    __syncthreads();

    // Standard parallel reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }
    const float s_m = s_mem[0];

    // --- 2. Compute our_kq[m, n] for n = 0..N-1 ---
    const vector_t* k_quant_zero_h2 = reinterpret_cast<const vector_t*>(k_quant_zero);
    const vector_t* k_quant_scale_h2 = reinterpret_cast<const vector_t*>(k_quant_scale);
    const vector_t* t_h2 = reinterpret_cast<const vector_t*>(t);
    vector_t* our_kq_h2 = reinterpret_cast<vector_t*>(our_kq);
    
    const int N_h2 = N / 2;

    for (int n_h2 = tid; n_h2 < N_h2; n_h2 += BLOCK_SIZE) {
        vector_t k_zero_val = k_quant_zero_h2[n_h2];
        vector_t t_val = t_h2[m * N_h2 + n_h2];

        float u1 = s_m * Traits::to_float(k_zero_val.x) + Traits::to_float(t_val.x);
        float u2 = s_m * Traits::to_float(k_zero_val.y) + Traits::to_float(t_val.y);
        
        vector_t k_scale_val = k_quant_scale_h2[n_h2];
        vector_t result;
        result.x = Traits::from_float(u1 * Traits::to_float(k_scale_val.x));
        result.y = Traits::from_float(u2 * Traits::to_float(k_scale_val.y));

        our_kq_h2[m * N_h2 + n_h2] = result;
    }
}

template<typename T>
float fused_kq_launcher_impl(
    torch::Tensor& our_kq,
    const torch::Tensor& q,
    const torch::Tensor& k_quant_zero,
    const torch::Tensor& k_quant_scale,
    const torch::Tensor& t
) {
    const int M = q.size(0);
    const int K = q.size(1);
    const int N = t.size(1);

    TORCH_CHECK(q.dim() == 2, "q must be a 2D tensor");
    TORCH_CHECK(t.dim() == 2, "t must be a 2D tensor");
    TORCH_CHECK(k_quant_zero.dim() == 1, "k_quant_zero must be a 1D tensor");
    TORCH_CHECK(k_quant_scale.dim() == 1, "k_quant_scale must be a 1D tensor");
    TORCH_CHECK(our_kq.dim() == 2, "our_kq must be a 2D tensor");

    TORCH_CHECK(K % 2 == 0, "K must be a multiple of 2 for vectorized kernel");
    TORCH_CHECK(N % 2 == 0, "N must be a multiple of 2 for vectorized kernel");
    TORCH_CHECK(t.size(0) == M, "t.size(0) must be equal to q.size(0)");
    TORCH_CHECK(k_quant_zero.size(0) == N, "k_quant_zero.size(0) must be equal to t.size(1)");
    TORCH_CHECK(k_quant_scale.size(0) == N, "k_quant_scale.size(0) must be equal to t.size(1)");
    TORCH_CHECK(our_kq.size(0) == M, "our_kq.size(0) must be equal to q.size(0)");
    TORCH_CHECK(our_kq.size(1) == N, "our_kq.size(1) must be equal to t.size(1)");

    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(t.is_cuda(), "t must be a CUDA tensor");
    TORCH_CHECK(k_quant_zero.is_cuda(), "k_quant_zero must be a CUDA tensor");
    TORCH_CHECK(k_quant_scale.is_cuda(), "k_quant_scale must be a CUDA tensor");
    TORCH_CHECK(our_kq.is_cuda(), "our_kq must be a CUDA tensor");

    T* our_kq_ptr = reinterpret_cast<T*>(our_kq.data_ptr());
    const T* q_ptr = reinterpret_cast<const T*>(q.data_ptr());
    const T* k_quant_zero_ptr = reinterpret_cast<const T*>(k_quant_zero.data_ptr());
    const T* k_quant_scale_ptr = reinterpret_cast<const T*>(k_quant_scale.data_ptr());
    const T* t_ptr = reinterpret_cast<const T*>(t.data_ptr());

    const int block_size = 256;
    const dim3 grid_dim(M);
    const dim3 block_dim(block_size);
    const int shared_mem_size = block_size * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    if (M == 40 && K == 128) {
        fused_kq_kernel<T, block_size, 40, 128><<<grid_dim, block_dim, shared_mem_size>>>(
            our_kq_ptr,
            q_ptr,
            k_quant_zero_ptr,
            k_quant_scale_ptr,
            t_ptr,
            N
        );
    } else if (M == 8 && K == 128) {
        fused_kq_kernel<T, block_size, 8, 128><<<grid_dim, block_dim, shared_mem_size>>>(
            our_kq_ptr,
            q_ptr,
            k_quant_zero_ptr,
            k_quant_scale_ptr,
            t_ptr,
            N
        );
    } else {
        throw std::runtime_error("Unsupported kernel configuration");
    }
    
    cudaEventRecord(stop);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    
    cudaEventSynchronize(stop);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel execution failed: ", cudaGetErrorString(err));
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

float fused_kq_launcher(
    torch::Tensor& our_kq,
    const torch::Tensor& q,
    const torch::Tensor& k_quant_zero,
    const torch::Tensor& k_quant_scale,
    const torch::Tensor& t
) {
    if (q.scalar_type() == at::ScalarType::Half) {
        return fused_kq_launcher_impl<__half>(our_kq, q, k_quant_zero, k_quant_scale, t);
    } else if (q.scalar_type() == at::ScalarType::BFloat16) {
        return fused_kq_launcher_impl<__nv_bfloat16>(our_kq, q, k_quant_zero, k_quant_scale, t);
    } else {
        throw std::runtime_error("Unsupported data type for fused_kq_launcher");
    }
}

template<typename T, int BLOCK_SIZE, int M_STATIC, int D_STATIC>
__global__
void fused_wv_kernel(
    T* __restrict__ our_wv,
    const T* __restrict__ w,
    const T* __restrict__ v_quant_zero,
    const T* __restrict__ v_quant_scale,
    const T* __restrict__ term1,
    const int N
) {
    using Traits = TypeTraits<T>;
    using scalar_t = typename Traits::scalar_t;
    using vector_t = typename Traits::vector_t;
    
    const int m = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float s_mem[];

    // --- 1. Compute wv_m = dot(w[m, :], v_temp) where v_temp = v_quant_zero * v_quant_scale ---
    const vector_t* w_row_h2 = reinterpret_cast<const vector_t*>(w + m * N);
    const vector_t* v_quant_zero_h2 = reinterpret_cast<const vector_t*>(v_quant_zero);
    const vector_t* v_quant_scale_h2 = reinterpret_cast<const vector_t*>(v_quant_scale);
    
    const int N_h2 = N / 2;
    float thread_sum = 0.0f;

    for (int n_h2 = tid; n_h2 < N_h2; n_h2 += BLOCK_SIZE) {
        vector_t w_val_h2 = w_row_h2[n_h2];
        vector_t v_zero_val_h2 = v_quant_zero_h2[n_h2];
        vector_t v_scale_val_h2 = v_quant_scale_h2[n_h2];

        float w1 = Traits::to_float(w_val_h2.x);
        float w2 = Traits::to_float(w_val_h2.y);
        float vz1 = Traits::to_float(v_zero_val_h2.x);
        float vz2 = Traits::to_float(v_zero_val_h2.y);
        float vs1 = Traits::to_float(v_scale_val_h2.x);
        float vs2 = Traits::to_float(v_scale_val_h2.y);
        
        thread_sum += w1 * vz1 * vs1;
        thread_sum += w2 * vz2 * vs2;
    }
    
    s_mem[tid] = thread_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }
    const float wv_m = s_mem[0];

    // --- 2. Compute our_wv[m, d] = wv_m + term1[m, d] for d = 0..D_STATIC-1 ---
    vector_t* our_wv_h2 = reinterpret_cast<vector_t*>(our_wv);
    const vector_t* term1_h2 = reinterpret_cast<const vector_t*>(term1);
    const int D_h2 = D_STATIC / 2;

    for (int d_h2 = tid; d_h2 < D_h2; d_h2 += BLOCK_SIZE) {
        const int term1_idx_h2 = m * D_h2 + d_h2;
        const int out_idx_h2 = m * D_h2 + d_h2;
        
        vector_t term1_val_h2 = term1_h2[term1_idx_h2];
        float t1 = Traits::to_float(term1_val_h2.x);
        float t2 = Traits::to_float(term1_val_h2.y);
        
        vector_t result_h2;
        result_h2.x = Traits::from_float(wv_m + t1);
        result_h2.y = Traits::from_float(wv_m + t2);
        
        our_wv_h2[out_idx_h2] = result_h2;
    }
}

template<typename T>
float fused_wv_launcher_impl(
    torch::Tensor& our_wv,
    const torch::Tensor& w,
    const torch::Tensor& v_quant_zero,
    const torch::Tensor& v_quant_scale,
    const torch::Tensor& term1
) {
    const int M = w.size(0);
    const int N = w.size(1);
    const int D = term1.size(1);

    TORCH_CHECK(w.dim() == 2, "w must be a 2D tensor");
    TORCH_CHECK(v_quant_zero.dim() == 1, "v_quant_zero must be a 1D tensor");
    TORCH_CHECK(v_quant_scale.dim() == 1, "v_quant_scale must be a 1D tensor");
    TORCH_CHECK(term1.dim() == 2, "term1 must be a 2D tensor");
    TORCH_CHECK(our_wv.dim() == 2, "our_wv must be a 2D tensor");

    TORCH_CHECK(N % 2 == 0, "N must be a multiple of 2 for vectorized kernel");
    TORCH_CHECK(D % 2 == 0, "D must be a multiple of 2 for vectorized kernel");
    TORCH_CHECK(term1.size(0) == M, "term1.size(0) must be equal to M");
    TORCH_CHECK(v_quant_zero.size(0) == N, "v_quant_zero.size(0) must be equal to N");
    TORCH_CHECK(v_quant_scale.size(0) == N, "v_quant_scale.size(0) must be equal to N");
    TORCH_CHECK(our_wv.size(0) == M, "our_wv.size(0) must be equal to M");
    TORCH_CHECK(our_wv.size(1) == D, "our_wv.size(1) must be equal to D");

    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(v_quant_zero.is_cuda(), "v_quant_zero must be a CUDA tensor");
    TORCH_CHECK(v_quant_scale.is_cuda(), "v_quant_scale must be a CUDA tensor");
    TORCH_CHECK(term1.is_cuda(), "term1 must be a CUDA tensor");
    TORCH_CHECK(our_wv.is_cuda(), "our_wv must be a CUDA tensor");

    T* our_wv_ptr = reinterpret_cast<T*>(our_wv.data_ptr());
    const T* w_ptr = reinterpret_cast<const T*>(w.data_ptr());
    const T* v_quant_zero_ptr = reinterpret_cast<const T*>(v_quant_zero.data_ptr());
    const T* v_quant_scale_ptr = reinterpret_cast<const T*>(v_quant_scale.data_ptr());
    const T* term1_ptr = reinterpret_cast<const T*>(term1.data_ptr());

    const int block_size = 256;
    const dim3 grid_dim(M);
    const dim3 block_dim(block_size);
    const int shared_mem_size = block_size * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    if (M == 40 && D == 128) {
        fused_wv_kernel<T, block_size, 40, 128><<<grid_dim, block_dim, shared_mem_size>>>(
            our_wv_ptr,
            w_ptr,
            v_quant_zero_ptr,
            v_quant_scale_ptr,
            term1_ptr,
            N
        );
    } else if (M == 8 && D == 128) {
        fused_wv_kernel<T, block_size, 8, 128><<<grid_dim, block_dim, shared_mem_size>>>(
            our_wv_ptr,
            w_ptr,
            v_quant_zero_ptr,
            v_quant_scale_ptr,
            term1_ptr,
            N
        );
    } else {
        throw std::runtime_error("Unsupported kernel configuration for fused_wv");
    }
    
    cudaEventRecord(stop);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    
    cudaEventSynchronize(stop);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel execution failed: ", cudaGetErrorString(err));
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

float fused_wv_launcher(
    torch::Tensor& our_wv,
    const torch::Tensor& w,
    const torch::Tensor& v_quant_zero,
    const torch::Tensor& v_quant_scale,
    const torch::Tensor& term1
) {
    if (w.scalar_type() == at::ScalarType::Half) {
        return fused_wv_launcher_impl<__half>(our_wv, w, v_quant_zero, v_quant_scale, term1);
    } else if (w.scalar_type() == at::ScalarType::BFloat16) {
        return fused_wv_launcher_impl<__nv_bfloat16>(our_wv, w, v_quant_zero, v_quant_scale, term1);
    } else {
        throw std::runtime_error("Unsupported data type for fused_wv_launcher");
    }
} 