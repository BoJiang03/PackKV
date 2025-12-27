#include <cuda_runtime.h>
#include <kernels.h>
#include <utils.h>
#include <type_traits.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#include <c10/core/ScalarType.h>

#define ctx_len_block_idx (blockIdx.x)
#define hidden_dim_block_idx (blockIdx.y)
#define ctx_len_block_num_cuda (gridDim.x)
#define hidden_dim_block_num_cuda (gridDim.y)
#define thread_idx (threadIdx.x)
#define warp_idx (threadIdx.x / WARP_THREAD_NUM)
#define in_warp_thread_idx (threadIdx.x % WARP_THREAD_NUM)

__device__ char * print_uint8_array(const uint8_t *array, int size) {
    // Each byte can be up to 3 digits (255) + 1 space, except the last one doesn't need space
    char *result = new char[size * 4];
    int pos = 0;
    
    for (int i = 0; i < size; i++) {
        uint8_t byte = array[i];
        
        // Convert byte to decimal string
        if (byte >= 100) {
            result[pos++] = '0' + (byte / 100);
            result[pos++] = '0' + ((byte / 10) % 10);
            result[pos++] = '0' + (byte % 10);
        } else if (byte >= 10) {
            result[pos++] = '0' + (byte / 10);
            result[pos++] = '0' + (byte % 10);
        } else {
            result[pos++] = '0' + byte;
        }
        
        // Add space between values (except for the last one)
        if (i < size - 1) {
            result[pos++] = ' ';
        }
    }
    
    result[pos] = '\0';
    return result;
}

__device__ char * print_uint64_t(uint64_t value) {
    char *result = new char[32];
    int pos = 0;
    
    for (int i = 0; i < 8; i++) {
        uint8_t byte = (value >> (i * 8)) & 0xFF;
        
        if (byte >= 100) {
            result[pos++] = '0' + (byte / 100);
            result[pos++] = '0' + ((byte / 10) % 10);
            result[pos++] = '0' + (byte % 10);
        } else if (byte >= 10) {
            result[pos++] = '0' + (byte / 10);
            result[pos++] = '0' + (byte % 10);
        } else {
            result[pos++] = '0' + byte;
        }
        
        if (i < 7) {
            result[pos++] = ' ';
        }
    }
    
    result[pos] = '\0';
    return result;
}

__device__ __forceinline__ void warp_load(
    const uint8_t* from,
    uint32_t* to,
    const size_t size
) {
    const int lane_id = in_warp_thread_idx;
    constexpr size_t warp_stride_bytes = 128;
    for (size_t offset = lane_id * 4; offset < size; offset += warp_stride_bytes) {
        memcpy(&to[offset / 4], from + offset, sizeof(uint32_t));
    }
}

__device__ __forceinline__ void warp_load_128(const uint8_t* from, uint32_t* to) {
    const int lane_id = in_warp_thread_idx;
    memcpy(&to[lane_id], from + lane_id * 4, sizeof(uint32_t));
}

__device__ __forceinline__ void warp_store_128(const uint32_t* from, uint32_t* to) {
    const int lane_id = in_warp_thread_idx;
    memcpy(to + lane_id, from + lane_id, sizeof(uint32_t));
}

template <size_t RAW_BIT_LEN, size_t PACK_PER_THREAD>
__device__ __forceinline__ uint16_t load_bits(const uint32_t* bits_buffer) {
    constexpr size_t BITS_PER_THREAD = RAW_BIT_LEN * PACK_PER_THREAD;
    
    static_assert(BITS_PER_THREAD <= 16, "BITS_PER_THREAD must be less than or equal to 16");

    const int lane_id = in_warp_thread_idx;

    if constexpr (RAW_BIT_LEN == 4) {
        const uint16_t* data = reinterpret_cast<const uint16_t*>(bits_buffer);
        return data[lane_id];

    } else if constexpr (RAW_BIT_LEN == 2) {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(bits_buffer);
        return static_cast<uint16_t>(data[lane_id]);

    } else if constexpr (RAW_BIT_LEN == 3) {
        const size_t start_bit_offset = lane_id * BITS_PER_THREAD;
        const size_t start_byte_index = start_bit_offset / 8;
        const size_t bit_offset_in_byte = start_bit_offset % 8;

        const uint8_t* byte_buffer = reinterpret_cast<const uint8_t*>(bits_buffer);

        // Perform a safe, byte-wise load to avoid misaligned access
        uint16_t packed_bits;
        memcpy(&packed_bits, byte_buffer + start_byte_index, sizeof(uint16_t));

        uint16_t shifted_bits = packed_bits >> bit_offset_in_byte;

        return shifted_bits & 0xFFF;
    } else {
        static_assert(RAW_BIT_LEN == 4 || RAW_BIT_LEN == 2 || RAW_BIT_LEN == 3, "Unsupported RAW_BIT_LEN");
        return 0;
    }
}

template<typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return TypeTraits<T>::add_scalar(a, b);
    }
};

__device__ __forceinline__ uint16_t warp_exclusive_scan(
    uint8_t value
) {
    typedef cub::WarpScan<uint32_t> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage;
    uint32_t output;
    WarpScan(temp_storage).ExclusiveSum(static_cast<uint32_t>(value), output);
    return static_cast<uint16_t>(output);
}

template <typename T, size_t RAW_BIT_LEN, size_t WARP_NUM, size_t PACK_PER_THREAD>
__device__ __forceinline__ typename TypeTraits<T>::scalar_t static_bit_len_warp_dot_product(
    const typename TypeTraits<T>::vector_t q_1_2, 
    const typename TypeTraits<T>::vector_t q_3_4, 
    const uint16_t mins, 
    const uint8_t round_idx) {
    static_assert(PACK_PER_THREAD == 4, "PACK_PER_THREAD must be 4");
    constexpr size_t MASK = (1 << RAW_BIT_LEN) - 1;

    using Traits = TypeTraits<T>;
    using scalar_t = typename Traits::scalar_t;
    using vector_t = typename Traits::vector_t;

    // if (hidden_dim_block_idx == 1 && round_idx == 0) {
    //     printf("round_idx: %d, in_warp_thread_idx: %d, mins:%d, %d, %d, %d\n", (int)round_idx, (int)in_warp_thread_idx, (int)(mins & MASK), (int)((mins >> RAW_BIT_LEN) & MASK), (int)((mins >> (RAW_BIT_LEN * 2)) & MASK), (int)((mins >> (RAW_BIT_LEN * 3)) & MASK));
    // }

    vector_t partial_dot = Traits::add(
        Traits::mul(q_1_2, Traits::make_vector(
            Traits::from_uint(mins & MASK),
            Traits::from_uint((mins >> RAW_BIT_LEN) & MASK)
        )),
        Traits::mul(q_3_4, Traits::make_vector(
            Traits::from_uint((mins >> (RAW_BIT_LEN * 2)) & MASK),
            Traits::from_uint((mins >> (RAW_BIT_LEN * 3)) & MASK)
        ))
    );

    scalar_t dot_product = Traits::vector_sum(partial_dot);

    scalar_t sum = Traits::warp_reduce_sum(dot_product);

    return sum;
}

template <size_t RAW_BIT_LEN, size_t HIDDEN_DIM_BLOCK_SIZE>
__device__ __forceinline__ uint16_t load_mins(
    const uint32_t* buffer
) {
    constexpr size_t PACK_PER_THREAD = HIDDEN_DIM_BLOCK_SIZE / WARP_THREAD_NUM;
    return load_bits<RAW_BIT_LEN, PACK_PER_THREAD>(buffer);
}

template <size_t RAW_BIT_LEN, size_t HIDDEN_DIM_BLOCK_SIZE, size_t PACK_PER_THREAD>
__device__ __forceinline__ void load_encode_lens(
    const uint32_t* buffer,
    uint8_t encode_lens[PACK_PER_THREAD]
) {
    const uint16_t packed_encode_lens = load_bits<RAW_BIT_LEN, PACK_PER_THREAD>(buffer);
    const uint8_t MASK = (1 << RAW_BIT_LEN) - 1;
    #pragma unroll
    for (int i = 0; i < PACK_PER_THREAD; i++) {
        encode_lens[i] = (packed_encode_lens >> (i * RAW_BIT_LEN)) & MASK;
    }
}

__device__ __forceinline__ uint64_t ld_shared_u64(const void* addr) {
    uint64_t val;
    asm volatile(
        "ld.shared.u64 %0, [%1];\n"
        : "=l"(val)
        : "l"(addr)
    );
    return val;
}

__device__ __forceinline__
void load_pack_from_buffer(
    const uint32_t* buffer,
    const uint16_t  pack_offset,
    const uint8_t   pack_size,
    uint64_t&       pack
) {
    const char* byte_ptr = reinterpret_cast<const char*>(buffer) + pack_offset;

    // asm volatile(
    //     "ld.shared.u64 %0, [%1];\n\t"
    //     : "=l"(pack)
    //     : "l"(byte_ptr)
    // );

    memcpy(&pack, byte_ptr, pack_size);
}


template <size_t RAW_BIT_LEN, size_t HIDDEN_DIM_BLOCK_SIZE, size_t PACK_PER_THREAD, size_t WARP_NUM>
__device__ __forceinline__ void load_packed_data(
    const uint32_t* buffer,
    const uint8_t encode_lens[PACK_PER_THREAD],
    uint64_t& pack1,
    uint64_t& pack2,
    uint64_t& pack3,
    uint64_t& pack4,
    const uint8_t round_idx
) {
    uint8_t pack_size = encode_lens[0] * PACK_ELE_NUM / 8;
    uint16_t pack_offset = warp_exclusive_scan(pack_size);

    load_pack_from_buffer(
        buffer,
        pack_offset,
        pack_size,
        pack1
    );

    pack_offset = __shfl_sync(0xFFFFFFFF, pack_offset + pack_size, WARP_THREAD_NUM - 1);
    pack_size = encode_lens[1] * PACK_ELE_NUM / 8;
    pack_offset += warp_exclusive_scan(pack_size);

    load_pack_from_buffer(
        buffer,
        pack_offset,
        pack_size,
        pack2
    );

    pack_offset = __shfl_sync(0xFFFFFFFF, pack_offset + pack_size, WARP_THREAD_NUM - 1);
    pack_size = encode_lens[2] * PACK_ELE_NUM / 8;
    pack_offset += warp_exclusive_scan(pack_size);

    load_pack_from_buffer(
        buffer,
        pack_offset,
        pack_size,
        pack3
    );

    pack_offset = __shfl_sync(0xFFFFFFFF, pack_offset + pack_size, WARP_THREAD_NUM - 1);
    pack_size = encode_lens[3] * PACK_ELE_NUM / 8;
    pack_offset += warp_exclusive_scan(pack_size);

    load_pack_from_buffer(
        buffer,
        pack_offset,
        pack_size,
        pack4
    );
}

template <typename T, size_t RAW_BIT_LEN, size_t HIDDEN_DIM_BLOCK_SIZE, size_t WARP_NUM>
__device__ __forceinline__ void kq_mat_vec_mul_round(
    const uint8_t *compressed_buffer_ptr,
    const uint16_t encode_size,
    const T *q_ptr,
    typename TypeTraits<T>::vector_t *kq_out,
    const uint8_t round_idx
) {
    using Traits = TypeTraits<T>;
    using scalar_t = typename Traits::scalar_t;
    using vector_t = typename Traits::vector_t;
    
    constexpr size_t MINS_BYTE_NUM_PER_ROUND = RAW_BIT_LEN * HIDDEN_DIM_BLOCK_SIZE / 8;
    constexpr size_t ENCODE_LEN_BYTE_NUM_PER_ROUND = min_bits_for_encode_len(RAW_BIT_LEN) * HIDDEN_DIM_BLOCK_SIZE / 8;
    constexpr size_t MAX_ENCODE_SIZE_PER_ROUND = HIDDEN_DIM_BLOCK_SIZE * PACK_ELE_NUM * RAW_BIT_LEN / 8;
    constexpr size_t PACK_PER_THREAD = HIDDEN_DIM_BLOCK_SIZE / WARP_THREAD_NUM;
    alignas(uint32_t) __shared__ uint8_t buffer[WARP_NUM][((MINS_BYTE_NUM_PER_ROUND + ENCODE_LEN_BYTE_NUM_PER_ROUND + MAX_ENCODE_SIZE_PER_ROUND) / sizeof(uint32_t)) * sizeof(uint32_t)];

    __shared__ scalar_t q_min_dot_product[WARP_NUM];

    vector_t q_1_2, q_3_4;
    
    q_1_2 = __ldg(&((vector_t *)q_ptr)[in_warp_thread_idx]);
    q_3_4 = __ldg(&((vector_t *)q_ptr)[in_warp_thread_idx + 32]);
    // load mins and encode_lens(and some extra bytes)
    static_assert(MINS_BYTE_NUM_PER_ROUND + ENCODE_LEN_BYTE_NUM_PER_ROUND <= 128, "meta data must be less than 128 bytes");
    warp_load_128(
        compressed_buffer_ptr,
        reinterpret_cast<uint32_t*>(buffer[warp_idx])
    );
    // load mins and compute mins * q
    const uint16_t mins = load_mins<RAW_BIT_LEN, HIDDEN_DIM_BLOCK_SIZE>(reinterpret_cast<const uint32_t*>(buffer[warp_idx]));

    q_min_dot_product[warp_idx] = static_bit_len_warp_dot_product<T, RAW_BIT_LEN, WARP_NUM, PACK_PER_THREAD>(q_1_2, q_3_4, mins, round_idx);

    __syncwarp();

    // extract encode_lens
    static_assert(MINS_BYTE_NUM_PER_ROUND % sizeof(uint32_t) == 0, "MINS_BYTE_NUM_PER_ROUND must be divisible by 4");
    uint8_t encode_lens[PACK_PER_THREAD];
    load_encode_lens<min_bits_for_encode_len(RAW_BIT_LEN), HIDDEN_DIM_BLOCK_SIZE, PACK_PER_THREAD>(reinterpret_cast<const uint32_t*>(buffer[warp_idx] + MINS_BYTE_NUM_PER_ROUND), encode_lens);

    constexpr size_t BUFFER_CAPACITY = ((MINS_BYTE_NUM_PER_ROUND + ENCODE_LEN_BYTE_NUM_PER_ROUND + MAX_ENCODE_SIZE_PER_ROUND) / sizeof(uint32_t)) * sizeof(uint32_t);
    constexpr size_t REMAINED_BYTE_NUM = 128 - (MINS_BYTE_NUM_PER_ROUND + ENCODE_LEN_BYTE_NUM_PER_ROUND);
    
    if (encode_size > REMAINED_BYTE_NUM) {
        warp_load(
                compressed_buffer_ptr + 128,
                reinterpret_cast<uint32_t*>(buffer[warp_idx] + 128),
                encode_size - REMAINED_BYTE_NUM
        );
    }

    // if (thread_idx == 0 && hidden_dim_block_idx == 14) {
    //     printf("ctx_len_block_idx: %d, hidden_dim_block_idx: %d, round_idx: %d, encode_size: %d\n", (int)ctx_len_block_idx, (int)hidden_dim_block_idx, (int)round_idx, (int)encode_size);
    // }

    // load packed data to registers
    uint64_t pack[4];
    load_packed_data<RAW_BIT_LEN, HIDDEN_DIM_BLOCK_SIZE, PACK_PER_THREAD, WARP_NUM>(reinterpret_cast<const uint32_t*>(buffer[warp_idx] + MINS_BYTE_NUM_PER_ROUND + ENCODE_LEN_BYTE_NUM_PER_ROUND), encode_lens, pack[0], pack[1], pack[2], pack[3], round_idx);

    uint8_t mask[4] = {
        (1 << encode_lens[0]) - 1,
        (1 << encode_lens[1]) - 1,
        (1 << encode_lens[2]) - 1,
        (1 << encode_lens[3]) - 1
    };

    uint8_t shift[4] = {
        0,
        0,
        0,
        0
    };

    // printf("round_idx: %d, min * q: %.1f\n", round_idx, __half2float(q_min_dot_product[warp_idx]));

    #pragma unroll
    for (uint8_t in_pack_ele_idx = 0; in_pack_ele_idx < PACK_ELE_NUM; in_pack_ele_idx+=2) {
        vector_t partial_dot1 = Traits::add(
            Traits::mul(q_1_2, Traits::make_vector(
                Traits::from_uint((pack[0] >> shift[0]) & mask[0]),
                Traits::from_uint((pack[1] >> shift[1]) & mask[1])
            )),
            Traits::mul(q_3_4, Traits::make_vector(
                Traits::from_uint((pack[2] >> shift[2]) & mask[2]),
                Traits::from_uint((pack[3] >> shift[3]) & mask[3])
            ))
        );

        scalar_t dot1 = Traits::vector_sum(partial_dot1);

        shift[0] += encode_lens[0];
        shift[1] += encode_lens[1];
        shift[2] += encode_lens[2];
        shift[3] += encode_lens[3];

        vector_t partial_dot2 = Traits::add(
            Traits::mul(q_1_2, Traits::make_vector(
                Traits::from_uint((pack[0] >> shift[0]) & mask[0]),
                Traits::from_uint((pack[1] >> shift[1]) & mask[1])
            )),
            Traits::mul(q_3_4, Traits::make_vector(
                Traits::from_uint((pack[2] >> shift[2]) & mask[2]),
                Traits::from_uint((pack[3] >> shift[3]) & mask[3])
            ))
        );

        scalar_t dot2 = Traits::vector_sum(partial_dot2);

        shift[0] += encode_lens[0];
        shift[1] += encode_lens[1];
        shift[2] += encode_lens[2];
        shift[3] += encode_lens[3];

        vector_t dot_1_2 = Traits::make_vector(dot2, dot1);

        vector_t total_dot = Traits::warp_reduce_sum_vector(dot_1_2);

        if (in_warp_thread_idx == 0) {
            total_dot = Traits::add(total_dot, Traits::make_vector(q_min_dot_product[warp_idx], q_min_dot_product[warp_idx]));
            // printf("%.1f, %.1f\n", __half2float(total_dot.x), __half2float(total_dot.y));
            kq_out[(round_idx * PACK_ELE_NUM + PACK_ELE_NUM - in_pack_ele_idx - 1) / 2] = total_dot;
        }
    }
    // if (in_warp_thread_idx == 0) {
    //     q_min_dot_product[warp_idx] = sum;
    //     printf("round_idx: %d, sum: %f\n", (int)round_idx, __half2float(q_min_dot_product[warp_idx]));
    // }

    // printf("warp_idx: %d, in_warp_thread_idx: %d, buffer[%d]: %d, buffer[%d]: %d, buffer[%d]: %d, buffer[%d]: %d\n", (int)warp_idx, (int)in_warp_thread_idx, (int)(in_warp_thread_idx * 4), (int)buffer[warp_idx][in_warp_thread_idx * 4], (int)(in_warp_thread_idx * 4 + 1), (int)buffer[warp_idx][in_warp_thread_idx * 4 + 1], (int)(in_warp_thread_idx * 4 + 2), (int)buffer[warp_idx][in_warp_thread_idx * 4 + 2], (int)(in_warp_thread_idx * 4 + 3), (int)buffer[warp_idx][in_warp_thread_idx * 4 + 3]);
}

template <typename T, size_t HIDDEN_DIM, size_t RAW_BIT_LEN, size_t HIDDEN_DIM_BLOCK_SIZE, size_t CTX_LEN_BLOCK_SIZE, size_t WARP_NUM>
__global__ void kq_mat_vec_mul_kernel(
    const uint8_t *__restrict__ compressed_buffer_ptr,
    const uint32_t *__restrict__ block_info_buffer_ptr,
    const T *__restrict__ q_ptr,
    T *__restrict__ kq_out_ptr
) {
    using Traits = TypeTraits<T>;
    using vector_t = typename Traits::vector_t;
    
    constexpr size_t ROUND_NUM = CTX_LEN_BLOCK_SIZE / PACK_ELE_NUM;
    constexpr size_t ROUND_SIZE_BYTE_NUM = sizeof(uint16_t) * ROUND_NUM;
    constexpr size_t HIDDEN_DIM_BLOCK_NUM = HIDDEN_DIM / HIDDEN_DIM_BLOCK_SIZE;
    constexpr size_t MIN_BYTE_NUM_PER_ROUND = RAW_BIT_LEN * HIDDEN_DIM_BLOCK_SIZE / 8;
    constexpr size_t ENCODE_LEN_BYTE_NUM_PER_ROUND = min_bits_for_encode_len(RAW_BIT_LEN) * HIDDEN_DIM_BLOCK_SIZE / 8;

    __shared__ uint32_t round_idx;
    __shared__ vector_t kq_out[CTX_LEN_BLOCK_SIZE / 2];
    uint16_t block_idx = ctx_len_block_idx * HIDDEN_DIM_BLOCK_NUM + hidden_dim_block_idx;
    uint16_t *round_size = (uint16_t *)(compressed_buffer_ptr + block_info_buffer_ptr[block_idx * 2]);

    if (thread_idx == 0) {
        round_idx = 0;
    }

    __syncthreads();
    // if (hidden_dim_block_idx != 14) {
    //         return;
    //     }
    //     if (thread_idx == 0) {
    //         printf("ctx_len_block_idx: %d, hidden_dim_block_idx: %d\n", (int)ctx_len_block_idx, (int)hidden_dim_block_idx);
    //     }

    while (true) {
        uint32_t selected_round_idx = 0;
        if (in_warp_thread_idx == 0) {
            selected_round_idx = atomicAdd(&round_idx, 1);
        }
        selected_round_idx = __shfl_sync(0xFFFFFFFF, selected_round_idx, 0);
        if (selected_round_idx >= ROUND_NUM) break;

        uint16_t round_offset = selected_round_idx == 0 ? 0 : round_size[selected_round_idx - 1];

        // if (in_warp_thread_idx == 0) {
        //     printf(
        //     "warp_idx: %d, block=(%d,%d) round_start_offset: %d, encode_size: %d, q_ptr_offset: %d, kq_out_offset: %d\n", 
        //     (int)warp_idx,
        //     (int)blockIdx.x, (int)blockIdx.y,
        //     (int)(block_info_buffer_ptr[block_idx * 2] + ROUND_SIZE_BYTE_NUM + round_offset), 
        //     (int)(round_size[selected_round_idx] - round_offset - MIN_BYTE_NUM_PER_ROUND - ENCODE_LEN_BYTE_NUM_PER_ROUND), 
        //     (int)hidden_dim_block_idx * HEAD_DIM, 
        //     (int)(ctx_len_block_num_cuda * hidden_dim_block_idx + ctx_len_block_idx) * CTX_LEN_BLOCK_SIZE
        //     );  
        // }

        // if (ctx_len_block_idx == 1 && in_warp_thread_idx == 0) {
        //     // printf("round_size[0]: %d, round_size[1]: %d, round_size[2]: %d, round_size[3]: %d\n", (int)round_size[0], (int)round_size[1], (int)round_size[2], (int)round_size[3]);
        //     printf("round_idx: %d, round_offset: %d, encode_size: %d, q_ptr_offset: %d, kq_out_offset: %d\n", (int)selected_round_idx, (int)round_offset, (int)(round_size[selected_round_idx] - round_offset - MIN_BYTE_NUM_PER_ROUND - ENCODE_LEN_BYTE_NUM_PER_ROUND), (int)(hidden_dim_block_idx * HEAD_DIM), (int)((ctx_len_block_num_cuda * hidden_dim_block_idx + ctx_len_block_idx) * CTX_LEN_BLOCK_SIZE));
        // }

        kq_mat_vec_mul_round<T, RAW_BIT_LEN, HIDDEN_DIM_BLOCK_SIZE, WARP_NUM>(
            reinterpret_cast<const uint8_t*>(round_size)
            + ROUND_SIZE_BYTE_NUM 
            + round_offset,

            round_size[selected_round_idx] - round_offset - MIN_BYTE_NUM_PER_ROUND - ENCODE_LEN_BYTE_NUM_PER_ROUND,

            q_ptr + hidden_dim_block_idx * HEAD_DIM,

            // kq_out_ptr + (ctx_len_block_num_cuda * hidden_dim_block_idx + ctx_len_block_idx) * CTX_LEN_BLOCK_SIZE,

            kq_out,

            selected_round_idx
        );
    }

    __syncthreads();

    // if (ctx_len_block_idx == 1 && thread_idx == 0) {
    //     for (int i = 0; i < CTX_LEN_BLOCK_SIZE / 2; i++) {
    //         printf("%.1f, %.1f\n", __half2float(kq_out[i].x), __half2float(kq_out[i].y));
    //     }
    // }

    if (warp_idx == 0) {
        reinterpret_cast<vector_t*>(kq_out_ptr + (ctx_len_block_num_cuda * hidden_dim_block_idx + ctx_len_block_idx) * CTX_LEN_BLOCK_SIZE)[in_warp_thread_idx] = kq_out[in_warp_thread_idx];
    }

    // most ituitive way, but no workload balance in block
    // for (uint32_t selected_round_idx = warp_idx; selected_round_idx < ALL_ROUND_NUM; selected_round_idx += WARP_NUM) {
        
    // }
    
    // const uint32_t round_idx = round_idx_and_offset >> 16;
    // const uint32_t offset = round_idx_and_offset & 0xFFFF;

    // constexpr size_t MINS_BYTE_NUM_PER_ROUND = RAW_BIT_LEN * HIDDEN_DIM_BLOCK_SIZE / 8;
    // constexpr size_t ENCODE_LEN_BYTE_NUM_PER_ROUND = min_bits_for_encode_len(RAW_BIT_LEN) * HIDDEN_DIM_BLOCK_SIZE / 8;

    // const uint8_t head_idx = hidden_dim_block_idx / HEAD_DIM;


    // half2 q_1_2, q_3_4;

    // if (thread_idx == 0) {
    //     printf("HIDDEN_DIM: %d, RAW_BIT_LEN: %d, HIDDEN_DIM_BLOCK_SIZE: %d, CTX_LEN_BLOCK_SIZE: %d, WARP_NUM: %d\n", (int)HIDDEN_DIM, (int)RAW_BIT_LEN, (int)HIDDEN_DIM_BLOCK_SIZE, (int)CTX_LEN_BLOCK_SIZE, (int)WARP_NUM);
    //     printf("MINS_BYTE_NUM_PER_ROUND: %d, ENCODE_LEN_BYTE_NUM_PER_ROUND: %d\n", (int)MINS_BYTE_NUM_PER_ROUND, (int)ENCODE_LEN_BYTE_NUM_PER_ROUND);
    // }
}

template<typename T>
__host__ float kq_mat_vec_mul_impl(
        const torch::Tensor &compressed_buffer,
        const torch::Tensor &block_info_buffer,
        const torch::Tensor &q,
        torch::Tensor &kq_out,
        const int ctx_len,
        const int hidden_dim,
        const int ctx_len_block_size,
        const int hidden_dim_block_size,
        const int bits_len
) {
    cudaSetDevice(compressed_buffer.device().index());
    const int ctx_len_block_num = ctx_len / ctx_len_block_size;
    const int hidden_dim_block_num = hidden_dim / hidden_dim_block_size;

    const uint8_t *compressed_buffer_ptr = compressed_buffer.data_ptr<uint8_t>();
    const uint32_t *block_info_buffer_ptr = block_info_buffer.data_ptr<uint32_t>();
    const T *q_ptr = reinterpret_cast<const T*>(q.data_ptr());
    T *kq_out_ptr = reinterpret_cast<T*>(kq_out.data_ptr());

    dim3 grid(ctx_len_block_num, hidden_dim_block_num);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (hidden_dim == 5120 && hidden_dim_block_size == 128 && ctx_len_block_size == 64 && bits_len == 4) {
        constexpr size_t WARP_NUM = 2;
        constexpr size_t HIDDEN_DIM = 5120;
        constexpr size_t RAW_BIT_LEN = 4;
        constexpr size_t HIDDEN_DIM_BLOCK_SIZE = 128;
        constexpr size_t CTX_LEN_BLOCK_SIZE = 64;
        constexpr size_t THREAD_NUM = WARP_NUM * WARP_THREAD_NUM;
        static_assert(CTX_LEN_BLOCK_SIZE % THREAD_NUM == 0, "CTX_LEN_BLOCK_SIZE must be divisible by THREAD_NUM");
        cudaEventRecord(start);
        kq_mat_vec_mul_kernel<T, HIDDEN_DIM, RAW_BIT_LEN, HIDDEN_DIM_BLOCK_SIZE, CTX_LEN_BLOCK_SIZE, WARP_NUM><<<grid, THREAD_NUM>>>(compressed_buffer_ptr, block_info_buffer_ptr, q_ptr, kq_out_ptr);
        cudaEventRecord(stop);
    } else if (hidden_dim == 1024 && hidden_dim_block_size == 128 && ctx_len_block_size == 64 && bits_len == 4) {
        constexpr size_t WARP_NUM = 2;
        constexpr size_t HIDDEN_DIM = 1024;
        constexpr size_t RAW_BIT_LEN = 4;
        constexpr size_t HIDDEN_DIM_BLOCK_SIZE = 128;
        constexpr size_t CTX_LEN_BLOCK_SIZE = 64;
        constexpr size_t THREAD_NUM = WARP_NUM * WARP_THREAD_NUM;
        static_assert(CTX_LEN_BLOCK_SIZE % THREAD_NUM == 0, "CTX_LEN_BLOCK_SIZE must be divisible by THREAD_NUM");
        cudaEventRecord(start);
        kq_mat_vec_mul_kernel<T, HIDDEN_DIM, RAW_BIT_LEN, HIDDEN_DIM_BLOCK_SIZE, CTX_LEN_BLOCK_SIZE, WARP_NUM><<<grid, THREAD_NUM>>>(compressed_buffer_ptr, block_info_buffer_ptr, q_ptr, kq_out_ptr);
        cudaEventRecord(stop);
    } else {
        std::cout << "no such case" << std::endl;
        return 0.0;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kq_mat_vec_mul_kernel (%s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaEventSynchronize(stop);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed during kq_mat_vec_mul_kernel execution (%s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

__host__ float kq_mat_vec_mul(
        const torch::Tensor &compressed_buffer,
        const torch::Tensor &block_info_buffer,
        const torch::Tensor &q,
        torch::Tensor &kq_out,
        const int ctx_len,
        const int hidden_dim,
        const int ctx_len_block_size,
        const int hidden_dim_block_size,
        const int bits_len
) {
    if (q.scalar_type() == at::ScalarType::Half) {
        return kq_mat_vec_mul_impl<__half>(compressed_buffer, block_info_buffer, q, kq_out, ctx_len, hidden_dim, ctx_len_block_size, hidden_dim_block_size, bits_len);
    } else if (q.scalar_type() == at::ScalarType::BFloat16) {
        return kq_mat_vec_mul_impl<__nv_bfloat16>(compressed_buffer, block_info_buffer, q, kq_out, ctx_len, hidden_dim, ctx_len_block_size, hidden_dim_block_size, bits_len);
    } else {
        std::cout << "Unsupported data type" << std::endl;
        return 0.0;
    }
}

template <typename T, size_t HIDDEN_DIM, size_t RAW_BIT_LEN, size_t HIDDEN_DIM_BLOCK_SIZE, size_t CTX_LEN_BLOCK_SIZE, size_t WARP_NUM = 2>
__global__ void wv_mat_vec_mul_kernel(
    const uint8_t *__restrict__ compressed_buffer_ptr,
    const uint32_t *__restrict__ block_info_buffer_ptr,
    const T *__restrict__ w_ptr,
    float *__restrict__ wv_out_ptr,
    const uint32_t ctx_len
) {
    using Traits = TypeTraits<T>;
    using scalar_t = typename Traits::scalar_t;
    using vector_t = typename Traits::vector_t;
    
    constexpr int HEAD_DIM = 128;
    static_assert(HIDDEN_DIM_BLOCK_SIZE == 64, "hidden_dim_block_size must be 64");
    static_assert(WARP_NUM == 2, "WARP_NUM must be 2");
    constexpr size_t ROUND_NUM_PER_WARP = 2; // hard coded for this work
    constexpr size_t PACK_PER_THREAD = 4; // 4 * 16 == 64
    constexpr size_t WARP_SIZE_BYTE_NUM = sizeof(uint16_t) * WARP_NUM * ROUND_NUM_PER_WARP;
    constexpr size_t MIN_BYTE_NUM_PER_ROUND = RAW_BIT_LEN * PACK_PER_THREAD * WARP_THREAD_NUM / 8;
    constexpr size_t ENCODE_LEN_BYTE_NUM_PER_ROUND = min_bits_for_encode_len(RAW_BIT_LEN) * PACK_PER_THREAD * WARP_THREAD_NUM / 8;
    static_assert(MIN_BYTE_NUM_PER_ROUND % sizeof(uint32_t) == 0, "MIN_BYTE_NUM_PER_ROUND must be divisible by sizeof(uint32_t)");
    static_assert(ENCODE_LEN_BYTE_NUM_PER_ROUND % sizeof(uint32_t) == 0, "ENCODE_LEN_BYTE_NUM_PER_ROUND must be divisible by sizeof(uint32_t)");
    constexpr size_t HIDDEN_DIM_BLOCK_NUM = HIDDEN_DIM / HIDDEN_DIM_BLOCK_SIZE;
    constexpr size_t MAX_ENCODED_BYTE_NUM_PER_ROUND = RAW_BIT_LEN * PACK_PER_THREAD * WARP_THREAD_NUM * PACK_ELE_NUM / 8;

    uint8_t head_idx = hidden_dim_block_idx * HIDDEN_DIM_BLOCK_SIZE / HEAD_DIM;
    // uint8_t in_head_hidden_dim_block_idx = hidden_dim_block_idx * HEAD_DIM_BLOCK_SIZE % HEAD_DIM / WARP_THREAD_NUM;

    __shared__ vector_t w[CTX_LEN_BLOCK_SIZE / 2];
    __shared__ vector_t wv_out[HIDDEN_DIM_BLOCK_SIZE / 2];
    alignas(uint32_t) __shared__ uint8_t round_buffer[WARP_NUM][(MIN_BYTE_NUM_PER_ROUND + ENCODE_LEN_BYTE_NUM_PER_ROUND + MAX_ENCODED_BYTE_NUM_PER_ROUND) / sizeof(uint32_t) * sizeof(uint32_t)];

    const uint16_t *warp_sizes = (uint16_t *)(compressed_buffer_ptr + block_info_buffer_ptr[(ctx_len_block_idx * HIDDEN_DIM_BLOCK_NUM + hidden_dim_block_idx) * 2]);
    // load w to shared memory
    warp_load_128(
            reinterpret_cast<const uint8_t *>(w_ptr) + (ctx_len * head_idx + CTX_LEN_BLOCK_SIZE * ctx_len_block_idx) * sizeof(scalar_t) + warp_idx * 128,
            reinterpret_cast<uint32_t*>(w + (warp_idx * 128) / sizeof(vector_t))
    );

    __syncthreads();

    vector_t accu_wv{};

    #pragma unroll
    for (int in_warp_round_idx = 0; in_warp_round_idx < ROUND_NUM_PER_WARP; in_warp_round_idx++) {
        uint16_t warp_offset = warp_idx == 0 && in_warp_round_idx == 0 ? 0 : warp_sizes[warp_idx * ROUND_NUM_PER_WARP + in_warp_round_idx - 1];
        uint16_t warp_size = warp_sizes[warp_idx * ROUND_NUM_PER_WARP + in_warp_round_idx] - warp_offset;
        warp_offset += WARP_SIZE_BYTE_NUM;

        // load mins and encode_lens to shared memory
        static_assert(MIN_BYTE_NUM_PER_ROUND + ENCODE_LEN_BYTE_NUM_PER_ROUND <= 128, "MIN_BYTE_NUM_PER_ROUND + ENCODE_LEN_BYTE_NUM_PER_ROUND must be 128");
        warp_load_128(
            reinterpret_cast<const uint8_t *>(warp_sizes) + warp_offset,
            reinterpret_cast<uint32_t*>(round_buffer[warp_idx])
        );

        // __syncwarp();

        uint64_t pack = 0;
        uint16_t mins = 0;
        uint16_t encode_lens = 0;

        mins = load_bits<RAW_BIT_LEN, PACK_PER_THREAD>(reinterpret_cast<const uint32_t*>(round_buffer[warp_idx]));
        encode_lens = load_bits<min_bits_for_encode_len(RAW_BIT_LEN), PACK_PER_THREAD>(reinterpret_cast<const uint32_t*>(round_buffer[warp_idx] + MIN_BYTE_NUM_PER_ROUND));

        if (warp_size > 128) {
            warp_load(
                reinterpret_cast<const uint8_t *>(warp_sizes) + warp_offset + 128,
                reinterpret_cast<uint32_t*>(round_buffer[warp_idx] + 128),
                warp_size - 128
            );
        }

        // __syncwarp();

        uint16_t pack_data_offset = MIN_BYTE_NUM_PER_ROUND + ENCODE_LEN_BYTE_NUM_PER_ROUND;

        #pragma unroll
        for (int in_round_pack_idx = 0; in_round_pack_idx < PACK_PER_THREAD; in_round_pack_idx++) {
            const uint8_t encode_len = encode_lens >> (in_round_pack_idx * min_bits_for_encode_len(RAW_BIT_LEN)) & ((1 << min_bits_for_encode_len(RAW_BIT_LEN)) - 1);
            const uint8_t pack_size = encode_len * PACK_ELE_NUM / 8;
            uint16_t in_warp_pack_offset = warp_exclusive_scan(pack_size);
            load_pack_from_buffer(
                reinterpret_cast<const uint32_t*>(round_buffer[warp_idx]),
                pack_data_offset + in_warp_pack_offset,
                pack_size,
                pack
            );
            pack_data_offset += __shfl_sync(0xFFFFFFFF, in_warp_pack_offset + pack_size, WARP_THREAD_NUM - 1);

            uint8_t min = mins >> (in_round_pack_idx * RAW_BIT_LEN) & ((1 << RAW_BIT_LEN) - 1);

            uint8_t shifts[2] = {
                encode_len * (PACK_ELE_NUM - 1),
                encode_len * (PACK_ELE_NUM - 2)
            };

            uint8_t mask = (1 << encode_len) - 1;
            
            uint8_t w_offset = in_warp_round_idx * 32 + in_round_pack_idx * 8;
            #pragma unroll
            for (uint8_t i = 0; i < PACK_ELE_NUM; i+=2) {
                uint8_t pack_data_1 = (pack >> shifts[0] & mask) + min;
                uint8_t pack_data_2 = (pack >> shifts[1] & mask) + min;

                accu_wv = Traits::add(accu_wv, Traits::mul(Traits::make_vector(Traits::from_uint(pack_data_1), Traits::from_uint(pack_data_2)), w[w_offset + i / 2]));
                shifts[0] -= encode_len * 2;
                shifts[1] -= encode_len * 2;
            }
        }
    }

    
    scalar_t final_val = Traits::vector_sum(accu_wv);
    reinterpret_cast<scalar_t*>(wv_out)[warp_idx * WARP_THREAD_NUM + in_warp_thread_idx] = final_val;

    __syncthreads();

    if (warp_idx == 0) {
        // Convert vector_t to two floats and accumulate separately
        vector_t result = reinterpret_cast<vector_t*>(wv_out)[in_warp_thread_idx];
        float val1 = Traits::to_float(result.x);
        float val2 = Traits::to_float(result.y);
        
        atomicAdd(
            wv_out_ptr + hidden_dim_block_idx * HIDDEN_DIM_BLOCK_SIZE + in_warp_thread_idx * 2,
            val1
        );
        atomicAdd(
            wv_out_ptr + hidden_dim_block_idx * HIDDEN_DIM_BLOCK_SIZE + in_warp_thread_idx * 2 + 1,
            val2
        );
    }
}

template<typename T>
float wv_mat_vec_mul_impl(
        const torch::Tensor &compressed_buffer,
        const torch::Tensor &block_info_buffer,
        const torch::Tensor &w,
        torch::Tensor &wv_out,
        const int ctx_len,
        const int hidden_dim,
        const int ctx_len_block_size,
        const int hidden_dim_block_size,
        const int bits_len
) {
    cudaSetDevice(compressed_buffer.device().index());
    const int ctx_len_block_num = ctx_len / ctx_len_block_size;
    const int hidden_dim_block_num = hidden_dim / hidden_dim_block_size;

    const uint8_t *compressed_buffer_ptr = compressed_buffer.data_ptr<uint8_t>();
    const uint32_t *block_info_buffer_ptr = block_info_buffer.data_ptr<uint32_t>();
    const T *w_ptr = reinterpret_cast<const T*>(w.data_ptr());
    float *wv_out_ptr = wv_out.data_ptr<float>();
    dim3 grid(ctx_len_block_num, hidden_dim_block_num);

    // std::cout << "wv_mat_vec_mul" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (hidden_dim == 5120 && hidden_dim_block_size == 64 && ctx_len_block_size == 128 && bits_len == 3) {
        constexpr size_t WARP_NUM = 2;
        constexpr size_t HIDDEN_DIM = 5120;
        constexpr size_t RAW_BIT_LEN = 3;
        constexpr size_t HIDDEN_DIM_BLOCK_SIZE = 64;
        constexpr size_t CTX_LEN_BLOCK_SIZE = 128;
        constexpr size_t THREAD_NUM = WARP_NUM * WARP_THREAD_NUM;
        cudaEventRecord(start);
        wv_mat_vec_mul_kernel<T, HIDDEN_DIM, RAW_BIT_LEN, HIDDEN_DIM_BLOCK_SIZE, CTX_LEN_BLOCK_SIZE, WARP_NUM><<<grid, THREAD_NUM>>>(compressed_buffer_ptr, block_info_buffer_ptr, w_ptr, wv_out_ptr, ctx_len);
        cudaEventRecord(stop);
    } else if (hidden_dim == 1024 && hidden_dim_block_size == 64 && ctx_len_block_size == 128 && bits_len == 3) {
        constexpr size_t WARP_NUM = 2;
        constexpr size_t HIDDEN_DIM = 1024;
        constexpr size_t RAW_BIT_LEN = 3;
        constexpr size_t HIDDEN_DIM_BLOCK_SIZE = 64;
        constexpr size_t CTX_LEN_BLOCK_SIZE = 128;
        constexpr size_t THREAD_NUM = WARP_NUM * WARP_THREAD_NUM;
        cudaEventRecord(start);
        wv_mat_vec_mul_kernel<T, HIDDEN_DIM, RAW_BIT_LEN, HIDDEN_DIM_BLOCK_SIZE, CTX_LEN_BLOCK_SIZE, WARP_NUM><<<grid, THREAD_NUM>>>(compressed_buffer_ptr, block_info_buffer_ptr, w_ptr, wv_out_ptr, ctx_len);
        cudaEventRecord(stop);
    } else {
        std::cout << "no such case" << std::endl;
        return 0.0;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch wv_mat_vec_mul_kernel (%s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaEventSynchronize(stop);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed during wv_mat_vec_mul_kernel execution (%s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

float wv_mat_vec_mul(
        const torch::Tensor &compressed_buffer,
        const torch::Tensor &block_info_buffer,
        const torch::Tensor &w,
        torch::Tensor &wv_out,
        const int ctx_len,
        const int hidden_dim,
        const int ctx_len_block_size,
        const int hidden_dim_block_size,
        const int bits_len
) {
    if (w.scalar_type() == at::ScalarType::Half) {
        return wv_mat_vec_mul_impl<__half>(compressed_buffer, block_info_buffer, w, wv_out, ctx_len, hidden_dim, ctx_len_block_size, hidden_dim_block_size, bits_len);
    } else if (w.scalar_type() == at::ScalarType::BFloat16) {
        return wv_mat_vec_mul_impl<__nv_bfloat16>(compressed_buffer, block_info_buffer, w, wv_out, ctx_len, hidden_dim, ctx_len_block_size, hidden_dim_block_size, bits_len);
    } else {
        std::cout << "Unsupported data type" << std::endl;
        return 0.0;
    }
}