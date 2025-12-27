#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <static_def.h>
#include <utils.h>

// Type traits for different floating point types
template<typename T>
struct TypeTraits {};

template<>
struct TypeTraits<__half> {
    using scalar_t = __half;
    using vector_t = __half2;
    using accumulator_t = __half;
    
    __device__ __forceinline__ static scalar_t from_uint(uint32_t val) {
        return __uint2half_rn(val);
    }
    
    __device__ __forceinline__ static vector_t make_vector(scalar_t x, scalar_t y) {
        return __halves2half2(x, y);
    }
    
    __device__ __forceinline__ static vector_t add(vector_t a, vector_t b) {
        return __hadd2(a, b);
    }
    
    __device__ __forceinline__ static vector_t mul(vector_t a, vector_t b) {
        return __hmul2(a, b);
    }
    
    __device__ __forceinline__ static scalar_t add_scalar(scalar_t a, scalar_t b) {
        return __hadd(a, b);
    }
    
    __device__ __forceinline__ static scalar_t mul_scalar(scalar_t a, scalar_t b) {
        return __hmul(a, b);
    }
    
    __device__ __forceinline__ static scalar_t vector_sum(vector_t v) {
        return __hadd(v.x, v.y);
    }
    
    __device__ __forceinline__ static scalar_t warp_reduce_sum(scalar_t val) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val = __hadd(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        return __shfl_sync(0xFFFFFFFF, val, 0);
    }
    
    __device__ __forceinline__ static vector_t warp_reduce_sum_vector(vector_t val) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val = __hadd2(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        return __shfl_sync(0xFFFFFFFF, val, 0);
    }
    
    __device__ __forceinline__ static float to_float(scalar_t val) {
        return __half2float(val);
    }
    
    __device__ __forceinline__ static scalar_t from_float(float val) {
        return __float2half(val);
    }
};

template<>
struct TypeTraits<__nv_bfloat16> {
    using scalar_t = __nv_bfloat16;
    using vector_t = __nv_bfloat162;
    using accumulator_t = __nv_bfloat16;
    
    __device__ __forceinline__ static scalar_t from_uint(uint32_t val) {
        return __uint2bfloat16_rn(val);
    }
    
    __device__ __forceinline__ static vector_t make_vector(scalar_t x, scalar_t y) {
        return __halves2bfloat162(x, y);
    }
    
    __device__ __forceinline__ static vector_t add(vector_t a, vector_t b) {
        return __hadd2(a, b);
    }
    
    __device__ __forceinline__ static vector_t mul(vector_t a, vector_t b) {
        return __hmul2(a, b);
    }
    
    __device__ __forceinline__ static scalar_t add_scalar(scalar_t a, scalar_t b) {
        return __hadd(a, b);
    }
    
    __device__ __forceinline__ static scalar_t mul_scalar(scalar_t a, scalar_t b) {
        return __hmul(a, b);
    }
    
    __device__ __forceinline__ static scalar_t vector_sum(vector_t v) {
        return __hadd(v.x, v.y);
    }
    
    __device__ __forceinline__ static scalar_t warp_reduce_sum(scalar_t val) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val = __hadd(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        return __shfl_sync(0xFFFFFFFF, val, 0);
    }
    
    __device__ __forceinline__ static vector_t warp_reduce_sum_vector(vector_t val) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val = __hadd2(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        return __shfl_sync(0xFFFFFFFF, val, 0);
    }
    
    __device__ __forceinline__ static float to_float(scalar_t val) {
        return __bfloat162float(val);
    }
    
    __device__ __forceinline__ static scalar_t from_float(float val) {
        return __float2bfloat16_rn(val);
    }
}; 