//
// Created by tut44803 on 2/27/25.
//

#include <cpu_compress.h>
#include <utils.h>
#include <iostream>

namespace v_cpu {

template<size_t ROUND_NUM, size_t HIDDEN_DIM_BLOCK_SIZE, size_t CTX_LEN_BLOCK_SIZE>
void fill_mins_and_encode_lens(
    const uint8_t k_encode_block[CTX_LEN_BLOCK_SIZE][HIDDEN_DIM_BLOCK_SIZE],
    uint8_t mins_[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    uint8_t maxs_[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    uint8_t encode_lens_[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE]
) {
    for (size_t round_idx = 0; round_idx < ROUND_NUM; round_idx++) {
        for (size_t in_round_pack_idx = 0; in_round_pack_idx < HIDDEN_DIM_BLOCK_SIZE; in_round_pack_idx ++) {

            mins_[round_idx][in_round_pack_idx] = k_encode_block[round_idx * PACK_ELE_NUM][in_round_pack_idx];
            maxs_[round_idx][in_round_pack_idx] = k_encode_block[round_idx * PACK_ELE_NUM][in_round_pack_idx];
            for (size_t i = 0; i < PACK_ELE_NUM; i++) {
                mins_[round_idx][in_round_pack_idx] = std::min(mins_[round_idx][in_round_pack_idx], k_encode_block[round_idx * PACK_ELE_NUM+i][in_round_pack_idx]);
                maxs_[round_idx][in_round_pack_idx] = std::max(maxs_[round_idx][in_round_pack_idx], k_encode_block[round_idx * PACK_ELE_NUM+i][in_round_pack_idx]);
            }
            encode_lens_[round_idx][in_round_pack_idx] = std::ceil(std::log2(maxs_[round_idx][in_round_pack_idx] - mins_[round_idx][in_round_pack_idx] + 1));

        }
    }
}

template<size_t ROUND_NUM, size_t HIDDEN_DIM_BLOCK_SIZE, size_t CTX_LEN_BLOCK_SIZE, size_t PACK_PER_THREAD>
void encode_packs(
    const uint8_t k_encode_block[CTX_LEN_BLOCK_SIZE][HIDDEN_DIM_BLOCK_SIZE],
    uint64_t encoded_packs_[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    uint8_t mins_[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    uint8_t encode_lens_[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE]
) {
    for (size_t round_idx = 0; round_idx < ROUND_NUM; round_idx++) {
        for (size_t in_round_pack_idx = 0; in_round_pack_idx < HIDDEN_DIM_BLOCK_SIZE; in_round_pack_idx ++) {
            uint8_t encode_len = encode_lens_[round_idx][in_round_pack_idx];
            uint64_t encoded_pack = 0;
            for (size_t i = 0; i < PACK_ELE_NUM; i++) {
                encoded_pack <<= encode_len;
                encoded_pack |= k_encode_block[round_idx * PACK_ELE_NUM+i][in_round_pack_idx] - mins_[round_idx][in_round_pack_idx];
            }
            encoded_packs_[round_idx][in_round_pack_idx] = encoded_pack;

        }
    }
}

template<size_t HIDDEN_DIM, size_t HIDDEN_DIM_BLOCK_SIZE, size_t CTX_LEN_BLOCK_SIZE>
void load_2d_block(
    const uint8_t *quant_ints_tensor_ptr,
    uint8_t k_encode_block[CTX_LEN_BLOCK_SIZE][HIDDEN_DIM_BLOCK_SIZE],
    const int ctx_len_block_idx,
    const int hidden_dim_block_idx
) {
    for (int i = 0; i < CTX_LEN_BLOCK_SIZE; i++) {
        for (int j = 0; j < HIDDEN_DIM_BLOCK_SIZE; j++) {
            k_encode_block[i][j] = quant_ints_tensor_ptr[(ctx_len_block_idx * CTX_LEN_BLOCK_SIZE + i) * HIDDEN_DIM + hidden_dim_block_idx * HIDDEN_DIM_BLOCK_SIZE + j];
        }
    }
}

template<size_t ROUND_NUM, size_t HIDDEN_DIM_BLOCK_SIZE, size_t PACK_PER_THREAD>
size_t v_encoded_packs_compact(
    const uint64_t encoded_packs[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    const uint8_t encode_lens[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    uint8_t *compact_array,
    uint8_t in_warp_round_idx,
    uint8_t warp_idx
) {
    size_t offset = 0;
    for (size_t pack_idx = 0; pack_idx < PACK_PER_THREAD; pack_idx++) {
        for (size_t thread_idx = 0; thread_idx < WARP_THREAD_NUM; thread_idx++) {
            uint64_t encoded_pack = encoded_packs[in_warp_round_idx * PACK_PER_THREAD + pack_idx][warp_idx * WARP_THREAD_NUM + thread_idx];
            uint8_t pack_byte_num = encode_lens[in_warp_round_idx * PACK_PER_THREAD + pack_idx][warp_idx * WARP_THREAD_NUM + thread_idx] * PACK_ELE_NUM / 8;

            for (size_t i = 0; i < pack_byte_num; i++) {
                compact_array[offset + i] = encoded_pack & 0xFF;
                encoded_pack >>= 8;
            }

            offset += pack_byte_num;
        }
    }
    return offset;
}

template<size_t RAW_BIT_LEN, size_t ROUND_NUM, size_t HIDDEN_DIM_BLOCK_SIZE, size_t COMPACT_ARRAY_SIZE, size_t PACK_PER_THREAD>
void v_bits_warp_compact(
    const uint8_t bits_array[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    uint8_t compact_array[COMPACT_ARRAY_SIZE],
    uint8_t in_warp_round_idx,
    uint8_t warp_idx
) {
    if constexpr (RAW_BIT_LEN == 3) {
        for (size_t thread_idx = 0; thread_idx < WARP_THREAD_NUM; thread_idx+=2) {
            uint16_t warp_thread1 = 0;
            uint16_t warp_thread2 = 0;
            for (size_t i = 0; i < PACK_PER_THREAD; i++) {
                warp_thread1 |= bits_array[in_warp_round_idx * PACK_PER_THREAD + i][warp_idx * WARP_THREAD_NUM + thread_idx] << (i * 3);
            }
            for (size_t i = 0; i < PACK_PER_THREAD; i++) {
                warp_thread2 |= bits_array[in_warp_round_idx * PACK_PER_THREAD + i][warp_idx * WARP_THREAD_NUM + thread_idx + 1] << (i * 3);
            }
            uint8_t two_warp_thread_bits[3];
            two_warp_thread_bits[0] = warp_thread1 & 0xFF;
            two_warp_thread_bits[1] = ((warp_thread1 >> 8) & 0x0F) | ((warp_thread2 << 4) & 0xF0);
            two_warp_thread_bits[2] = (warp_thread2 >> 4) & 0xFF;
            for (size_t i = 0; i < 3; i++) {
                compact_array[thread_idx * (PACK_PER_THREAD * 3) / 8 + i] = two_warp_thread_bits[i];
            }
        }
    } else {
        for (size_t thread_idx = 0; thread_idx < WARP_THREAD_NUM; thread_idx++) {
            uint16_t warp_thread = 0;
            for (size_t i = 0; i < PACK_PER_THREAD; i++) {
                warp_thread |= bits_array[in_warp_round_idx * PACK_PER_THREAD + i][warp_idx * WARP_THREAD_NUM + thread_idx] << (i * RAW_BIT_LEN);
            }
            constexpr size_t COMPACT_BYTE_NUM = RAW_BIT_LEN * PACK_PER_THREAD / 8;
            static_assert(COMPACT_BYTE_NUM <= 2, "COMPACT_BYTE_NUM must be less than or equal to 2");
            for (size_t i = 0; i < COMPACT_BYTE_NUM; i++) {
                compact_array[thread_idx * COMPACT_BYTE_NUM + i] = warp_thread & 0xFF;
                warp_thread >>= 8;
            }
        }
    }
}

template<size_t RAW_BIT_LEN, size_t ROUND_NUM, size_t CTX_LEN_BLOCK_SIZE, size_t HIDDEN_DIM_BLOCK_SIZE, size_t PACK_PER_THREAD>
size_t v_data_compact(
    uint8_t mins[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    uint8_t encode_lens[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    uint64_t encoded_packs[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    uint8_t encode_data[CTX_LEN_BLOCK_SIZE * HIDDEN_DIM_BLOCK_SIZE]
) {
    // constexpr size_t BLOCK_MIN_BYTE_NUM = RAW_BIT_LEN * (ROUND_NUM * HIDDEN_DIM_BLOCK_SIZE) / 8;
    // constexpr size_t BITS_FOR_ENCODE_LEN = min_bits_for_encode_len(RAW_BIT_LEN);
    // constexpr size_t ENCODE_LEN_BYTE_NUM = BITS_FOR_ENCODE_LEN * (ROUND_NUM * HIDDEN_DIM_BLOCK_SIZE) / 8;
    constexpr size_t WARP_NUM = HIDDEN_DIM_BLOCK_SIZE / WARP_THREAD_NUM;
    constexpr size_t ROUND_PER_WARP = 2; // fix for this work
    constexpr size_t WARP_SIZE_BYTE_NUM = sizeof(uint16_t) * WARP_NUM * ROUND_PER_WARP;
    constexpr size_t MIN_BYTE_NUM_PER_ROUND_PER_WARP = PACK_PER_THREAD * RAW_BIT_LEN * WARP_THREAD_NUM / 8;
    constexpr size_t ENCODE_LEN_BYTE_NUM_PER_ROUND_PER_WARP = min_bits_for_encode_len(RAW_BIT_LEN) * PACK_PER_THREAD * WARP_THREAD_NUM / 8;

    // uint16_t round_mins_[ROUND_NUM][WARP_THREAD_NUM];
    // uint16_t round_encode_lens_[ROUND_NUM][WARP_THREAD_NUM];
    uint16_t warp_size[WARP_NUM * ROUND_PER_WARP] = {0};

    size_t offset = WARP_SIZE_BYTE_NUM;
    for (size_t warp_idx = 0; warp_idx < WARP_NUM; warp_idx++) {
        // std::cout << "round_idx: " << round_idx << std::endl;
        v_bits_warp_compact<RAW_BIT_LEN, ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, MIN_BYTE_NUM_PER_ROUND_PER_WARP, PACK_PER_THREAD>(mins, encode_data + offset, 0, warp_idx);
        offset += MIN_BYTE_NUM_PER_ROUND_PER_WARP;
        v_bits_warp_compact<min_bits_for_encode_len(RAW_BIT_LEN), ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, ENCODE_LEN_BYTE_NUM_PER_ROUND_PER_WARP, PACK_PER_THREAD>(encode_lens, encode_data + offset, 0, warp_idx);
        offset += ENCODE_LEN_BYTE_NUM_PER_ROUND_PER_WARP;
        size_t encode_size = v_encoded_packs_compact<ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, PACK_PER_THREAD>(encoded_packs, encode_lens, encode_data + offset, 0, warp_idx);
        offset += encode_size;
        warp_size[warp_idx * ROUND_PER_WARP] += encode_size + MIN_BYTE_NUM_PER_ROUND_PER_WARP + ENCODE_LEN_BYTE_NUM_PER_ROUND_PER_WARP;
        // std::cout << "warp_idx: " << warp_idx << "in_warp_round_idx: " << 0 << ", encode_size: " << encode_size << std::endl;
        v_bits_warp_compact<RAW_BIT_LEN, ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, MIN_BYTE_NUM_PER_ROUND_PER_WARP, PACK_PER_THREAD>(mins, encode_data + offset, 1, warp_idx);
        offset += MIN_BYTE_NUM_PER_ROUND_PER_WARP;
        v_bits_warp_compact<min_bits_for_encode_len(RAW_BIT_LEN), ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, ENCODE_LEN_BYTE_NUM_PER_ROUND_PER_WARP, PACK_PER_THREAD>(encode_lens, encode_data + offset, 1, warp_idx);
        offset += ENCODE_LEN_BYTE_NUM_PER_ROUND_PER_WARP;
        encode_size = v_encoded_packs_compact<ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, PACK_PER_THREAD>(encoded_packs, encode_lens, encode_data + offset, 1, warp_idx);
        offset += encode_size;
        // std::cout << "warp_idx: " << warp_idx << "in_warp_round_idx: " << 1 << ", encode_size: " << encode_size << std::endl;
        warp_size[warp_idx * ROUND_PER_WARP + 1] += encode_size + MIN_BYTE_NUM_PER_ROUND_PER_WARP + ENCODE_LEN_BYTE_NUM_PER_ROUND_PER_WARP;
        // std::cout << "warp_idx: " << warp_idx << " warp_size: " << warp_size[warp_idx] << std::endl;
    }

    for (size_t round_idx = 1; round_idx < ROUND_PER_WARP * WARP_NUM; round_idx++) {
        warp_size[round_idx] += warp_size[round_idx - 1];
        // printf("round_idx: %d, warp_size: %d\n", (int)round_idx, (int)warp_size[round_idx]);
    }

    for (size_t round_idx = 0; round_idx < ROUND_PER_WARP * WARP_NUM; round_idx++) {
        // printf("round_idx: %d, warp_size: %d\n", (int)round_idx, (int)warp_size[round_idx]);
        *reinterpret_cast<uint16_t *>(encode_data + round_idx * sizeof(uint16_t)) = warp_size[round_idx];
    }

    return offset;
}

template<size_t RAW_BIT_LEN, size_t HIDDEN_DIM_BLOCK_SIZE, size_t CTX_LEN_BLOCK_SIZE, size_t PACK_PER_THREAD>
size_t v_block_encode_cpu(
    const uint8_t k_encode_block[CTX_LEN_BLOCK_SIZE][HIDDEN_DIM_BLOCK_SIZE],
    uint8_t *compressed_buffer_ptr
) {
    constexpr size_t ROUND_NUM = CTX_LEN_BLOCK_SIZE / PACK_ELE_NUM;
    // constexpr size_t pack_num_ce = round_num_ce * HIDDEN_DIM_BLOCK_SIZE;

    uint8_t mins_[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE] = {0};
    uint8_t maxs_[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE] = {0};
    uint8_t encode_lens_[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE] = {0};
    // fill mins_ and encode_lens_
    fill_mins_and_encode_lens<ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, CTX_LEN_BLOCK_SIZE>(k_encode_block, mins_, maxs_, encode_lens_);
    // print mins_ and encode_lens_
    // for (size_t round_idx = 0; round_idx < ROUND_NUM; round_idx++) {
    //     for (size_t in_round_pack_idx = 0; in_round_pack_idx < HIDDEN_DIM_BLOCK_SIZE; in_round_pack_idx ++) {
    //         std::cout << (int)mins_[round_idx][in_round_pack_idx] << ", ";
    //     }
    //     std::cout << std::endl;
    // }
    //
    // for (size_t round_idx = 0; round_idx < ROUND_NUM; round_idx++) {
    //     for (size_t in_round_pack_idx = 0; in_round_pack_idx < HIDDEN_DIM_BLOCK_SIZE; in_round_pack_idx ++) {
    //         std::cout << (int)encode_lens_[round_idx][in_round_pack_idx] << ", ";
    //     }
    //     std::cout << std::endl;
    // }

    uint64_t encoded_packs_[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE] = {0};
    static_assert(RAW_BIT_LEN * PACK_ELE_NUM <= 64, "Max encoded size for one pack must be less than or equal to 64 bits");
    encode_packs<ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, CTX_LEN_BLOCK_SIZE, PACK_PER_THREAD>(k_encode_block, encoded_packs_, mins_, encode_lens_);

    uint8_t encode_data[CTX_LEN_BLOCK_SIZE * HIDDEN_DIM_BLOCK_SIZE] = {0};
    size_t final_data_size = v_data_compact<RAW_BIT_LEN, ROUND_NUM, CTX_LEN_BLOCK_SIZE, HIDDEN_DIM_BLOCK_SIZE, PACK_PER_THREAD>(mins_, encode_lens_, encoded_packs_, encode_data);

    copy_data(encode_data, compressed_buffer_ptr, final_data_size);

    return final_data_size;
}

template<size_t RAW_BIT_LEN, size_t HIDDEN_DIM, size_t HIDDEN_DIM_BLOCK_SIZE, size_t CTX_LEN_BLOCK_SIZE>
size_t v_encode_cpu(
    const uint8_t *quant_ints_tensor_ptr,
    uint8_t *compressed_buffer_ptr,
    uint32_t *block_info_buffer_ptr,
    size_t ctx_len
) {
    static_assert(RAW_BIT_LEN <= 8, "RAW_BIT_LEN must be less than or equal to 8");
    static_assert(CTX_LEN_BLOCK_SIZE % PACK_ELE_NUM == 0, "CTX_LEN_BLOCK_SIZE must be divisible by PACK_ELE_NUM");
    static_assert(PACK_ELE_NUM * RAW_BIT_LEN <= 64, "Max encoded size for one pack must be less than or equal to 64 bits");

    int ctx_len_block_num = ctx_len / CTX_LEN_BLOCK_SIZE;
    int hidden_dim_block_num = HIDDEN_DIM / HIDDEN_DIM_BLOCK_SIZE;

    uint8_t k_encode_block[CTX_LEN_BLOCK_SIZE][HIDDEN_DIM_BLOCK_SIZE] = {0};

    size_t compressed_size = 0;
    for (int ctx_len_block_idx = 0; ctx_len_block_idx < ctx_len_block_num; ctx_len_block_idx++) {
        for (int hidden_dim_block_idx = 0; hidden_dim_block_idx < hidden_dim_block_num; hidden_dim_block_idx++) {
            size_t linear_idx = ctx_len_block_idx * hidden_dim_block_num + hidden_dim_block_idx;
            load_2d_block<HIDDEN_DIM, HIDDEN_DIM_BLOCK_SIZE, CTX_LEN_BLOCK_SIZE>(quant_ints_tensor_ptr, k_encode_block, ctx_len_block_idx, hidden_dim_block_idx);
            size_t block_compressed_size = v_block_encode_cpu<RAW_BIT_LEN, HIDDEN_DIM_BLOCK_SIZE, CTX_LEN_BLOCK_SIZE, 4>(k_encode_block, compressed_buffer_ptr + compressed_size);
            block_info_buffer_ptr[linear_idx * 2] = compressed_size;
            // std::cout << ctx_len_block_idx << " " << hidden_dim_block_idx << " " << compressed_size << " " << block_compressed_size << std::endl;
            compressed_size += block_compressed_size;
            block_info_buffer_ptr[linear_idx * 2 + 1] = block_compressed_size;
        }
    }

    return compressed_size;
}

}

void v_encode_cpu_assert(
    const torch::Tensor &quant_ints_tensor,
    const torch::Tensor &compressed_buffer,
    const torch::Tensor &block_info_buffer,
    const int ctx_len,
    const int hidden_dim,
    const int ctx_len_block_size,
    const int hidden_dim_block_size,
    const int bits_len
) {
    // assert device cpu
    packkv_assert(quant_ints_tensor.device().is_cpu(), "quant_ints_tensor must be a CPU tensor");
    packkv_assert(compressed_buffer.device().is_cpu(), "compressed_buffer must be a CPU tensor");
    packkv_assert(block_info_buffer.device().is_cpu(), "block_info_buffer must be a CPU tensor");
    int ctx_len_t_ = quant_ints_tensor.size(0);
    int hidden_dim_t_ = quant_ints_tensor.size(1);
    packkv_assert(ctx_len_t_ == ctx_len, "ctx_len mismatch");
    packkv_assert(hidden_dim_t_ == hidden_dim, "hidden_dim mismatch");

    packkv_assert(ctx_len % ctx_len_block_size == 0, "ctx_len must be divisible by ctx_len_block_size");
    packkv_assert(hidden_dim % hidden_dim_block_size == 0, "hidden_dim must be divisible by hidden_dim_block_size");

    int block_num = (ctx_len / ctx_len_block_size) * (hidden_dim / hidden_dim_block_size);
    packkv_assert(block_info_buffer.size(0) == block_num, "block_info_buffer block num mismatch");
    packkv_assert(block_info_buffer.size(1) == 2, "block_info_buffer.size(1) must be 2");

    packkv_assert(bits_len <= 4, "bits_len must be less than or equal to 4");

    packkv_assert(ctx_len_block_size % PACK_ELE_NUM == 0, "ctx_len_block_size must be divisible by PACK_ELE_NUM");
//    packkv_assert(hidden_dim_block_size == 128, "hidden_dim_block_size must be 128, because this is most common head dim");
    packkv_assert(ctx_len_block_size > 0, "ctx_len_block_size must be greater than 0");
    packkv_assert(hidden_dim_block_size > 0, "hidden_dim_block_size must be greater than 0");
    packkv_assert(quant_ints_tensor.dtype() == torch::kUInt8, "quant_ints_tensor must be a UInt8 tensor");
    packkv_assert(compressed_buffer.dtype() == torch::kUInt8, "compressed_buffer must be a UInt8 tensor");
    packkv_assert(compressed_buffer.numel() > 0, "compressed_buffer must be a non-empty tensor");
    packkv_assert(block_info_buffer.dtype() == torch::kUInt32, "block_info_buffer must be a UInt32 tensor");
}

size_t v_encode_cpu_pyi(
    const torch::Tensor &quant_ints_tensor,
    torch::Tensor &compressed_buffer,
    torch::Tensor &block_info_buffer,
    const int ctx_len,
    const int hidden_dim,
    const int ctx_len_block_size,
    const int hidden_dim_block_size,
    const int bits_len
) {
    v_encode_cpu_assert(quant_ints_tensor, compressed_buffer, block_info_buffer, ctx_len, hidden_dim, ctx_len_block_size, hidden_dim_block_size, bits_len);

    const uint8_t *quant_ints_tensor_ptr = quant_ints_tensor.data_ptr<uint8_t>();
    uint8_t *compressed_buffer_ptr = compressed_buffer.data_ptr<uint8_t>();
    uint32_t *block_info_buffer_ptr = block_info_buffer.data_ptr<uint32_t>();
    const int ctx_len_block_num = ctx_len / ctx_len_block_size;
    const int hidden_dim_block_num = hidden_dim / hidden_dim_block_size;

    if (
        ctx_len_block_size == 128 &&
        hidden_dim_block_size == 64 &&
        hidden_dim == 5120 &&
        bits_len == 3
    ){
        return v_cpu::v_encode_cpu<3, 5120, 64, 128>(
            quant_ints_tensor_ptr,
            compressed_buffer_ptr,
            block_info_buffer_ptr,
            ctx_len
        );
    } else if (
        ctx_len_block_size == 128 &&
        hidden_dim_block_size == 64 &&
        hidden_dim == 1024 &&
        bits_len == 3
    ){
        return v_cpu::v_encode_cpu<3, 1024, 64, 128>(
            quant_ints_tensor_ptr,
            compressed_buffer_ptr,
            block_info_buffer_ptr,
            ctx_len
        );
    } else {
        throw std::runtime_error("not supported");
    }
}