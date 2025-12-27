//
// Created by tut44803 on 2/27/25.
//

#include <cpu_compress.h>
#include <utils.h>
#include <iostream>

namespace v_cpu {

template<size_t ROUND_NUM, size_t HIDDEN_DIM_BLOCK_SIZE, size_t CTX_LEN_BLOCK_SIZE, size_t PACK_PER_THREAD>
void decode_packs(
    uint8_t k_encode_block[CTX_LEN_BLOCK_SIZE][HIDDEN_DIM_BLOCK_SIZE],
    const uint64_t encoded_packs_[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    const uint8_t mins_[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    const uint8_t encode_lens_[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE]
) {
    for (size_t round_idx = 0; round_idx < ROUND_NUM; round_idx++) {
        for (size_t in_round_pack_idx = 0; in_round_pack_idx < HIDDEN_DIM_BLOCK_SIZE; in_round_pack_idx++) {
            uint8_t encode_len = encode_lens_[round_idx][in_round_pack_idx];
            uint64_t encoded_pack = encoded_packs_[round_idx][in_round_pack_idx];
            uint8_t min_val = mins_[round_idx][in_round_pack_idx];
            uint64_t mask = (1ULL << encode_len) - 1;

            for (int i = PACK_ELE_NUM - 1; i >= 0; i--) {
                uint8_t val = encoded_pack & mask;
                k_encode_block[round_idx * PACK_ELE_NUM + i][in_round_pack_idx] = val + min_val;
                encoded_pack >>= encode_len;
            }
        }
    }
}


template<size_t HIDDEN_DIM, size_t HIDDEN_DIM_BLOCK_SIZE, size_t CTX_LEN_BLOCK_SIZE>
void store_2d_block(
    uint8_t *quant_ints_tensor_ptr,
    const uint8_t k_encode_block[CTX_LEN_BLOCK_SIZE][HIDDEN_DIM_BLOCK_SIZE],
    const int ctx_len_block_idx,
    const int hidden_dim_block_idx
) {
    for (int i = 0; i < CTX_LEN_BLOCK_SIZE; i++) {
        for (int j = 0; j < HIDDEN_DIM_BLOCK_SIZE; j++) {
            quant_ints_tensor_ptr[(ctx_len_block_idx * CTX_LEN_BLOCK_SIZE + i) * HIDDEN_DIM + hidden_dim_block_idx * HIDDEN_DIM_BLOCK_SIZE + j] = k_encode_block[i][j];
        }
    }
}

template<size_t ROUND_NUM, size_t HIDDEN_DIM_BLOCK_SIZE, size_t PACK_PER_THREAD>
size_t v_encoded_packs_uncompact(
    uint64_t encoded_packs[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    const uint8_t encode_lens[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    const uint8_t *compact_array,
    uint8_t in_warp_round_idx,
    uint8_t warp_idx
) {
    size_t offset = 0;
    for (size_t pack_idx = 0; pack_idx < PACK_PER_THREAD; pack_idx++) {
        for (size_t thread_idx = 0; thread_idx < WARP_THREAD_NUM; thread_idx++) {
            uint8_t pack_byte_num = encode_lens[in_warp_round_idx * PACK_PER_THREAD + pack_idx][warp_idx * WARP_THREAD_NUM + thread_idx] * PACK_ELE_NUM / 8;
            uint64_t encoded_pack = 0;
            for (size_t i = 0; i < pack_byte_num; i++) {
                encoded_pack |= (uint64_t)(compact_array[offset + i]) << (i * 8);
            }
            encoded_packs[in_warp_round_idx * PACK_PER_THREAD + pack_idx][warp_idx * WARP_THREAD_NUM + thread_idx] = encoded_pack;
            offset += pack_byte_num;
        }
    }
    return offset;
}

template<size_t RAW_BIT_LEN, size_t ROUND_NUM, size_t HIDDEN_DIM_BLOCK_SIZE, size_t COMPACT_ARRAY_SIZE, size_t PACK_PER_THREAD>
void v_bits_warp_uncompact(
    const uint8_t compact_array[COMPACT_ARRAY_SIZE],
    uint8_t bits_array[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    uint8_t in_warp_round_idx,
    uint8_t warp_idx
) {
    if constexpr (RAW_BIT_LEN == 3) {
        for (size_t thread_idx = 0; thread_idx < WARP_THREAD_NUM; thread_idx += 2) {
            uint8_t two_warp_thread_bits[3];
            for (size_t i = 0; i < 3; i++) {
                two_warp_thread_bits[i] = compact_array[thread_idx * PACK_PER_THREAD * 3 / 8 + i];
            }
            uint16_t warp_thread1 = two_warp_thread_bits[0] | ((two_warp_thread_bits[1] & 0x0F) << 8);
            uint16_t warp_thread2 = ((two_warp_thread_bits[1] & 0xF0) >> 4) | (two_warp_thread_bits[2] << 4);

            for (size_t i = 0; i < PACK_PER_THREAD; i++) {
                bits_array[in_warp_round_idx * PACK_PER_THREAD + i][warp_idx * WARP_THREAD_NUM + thread_idx] = (warp_thread1 >> (i * 3)) & 0x07;
            }
            for (size_t i = 0; i < PACK_PER_THREAD; i++) {
                bits_array[in_warp_round_idx * PACK_PER_THREAD + i][warp_idx * WARP_THREAD_NUM + thread_idx + 1] = (warp_thread2 >> (i * 3)) & 0x07;
            }
        }
    } else {
        for (size_t thread_idx = 0; thread_idx < WARP_THREAD_NUM; thread_idx++) {
            constexpr size_t COMPACT_BYTE_NUM = RAW_BIT_LEN * PACK_PER_THREAD / 8;
            static_assert(COMPACT_BYTE_NUM <= 4, "COMPACT_BYTE_NUM must be less than or equal to 4");
            uint16_t warp_thread = 0;
            for (size_t i = 0; i < COMPACT_BYTE_NUM; i++) {
                warp_thread |= (uint32_t)compact_array[thread_idx * COMPACT_BYTE_NUM + i] << (i * 8);
            }
            for (size_t i = 0; i < PACK_PER_THREAD; i++) {
                bits_array[in_warp_round_idx * PACK_PER_THREAD + i][warp_idx * WARP_THREAD_NUM + thread_idx] = (warp_thread >> (i * RAW_BIT_LEN)) & ((1 << RAW_BIT_LEN) - 1);
            }
        }
    }
}

template<size_t RAW_BIT_LEN, size_t ROUND_NUM, size_t CTX_LEN_BLOCK_SIZE, size_t HIDDEN_DIM_BLOCK_SIZE, size_t PACK_PER_THREAD>
void v_data_uncompact(
    uint8_t mins[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    uint8_t encode_lens[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    uint64_t encoded_packs[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE],
    const uint8_t encode_data[CTX_LEN_BLOCK_SIZE * HIDDEN_DIM_BLOCK_SIZE]
) {
    constexpr size_t WARP_NUM = HIDDEN_DIM_BLOCK_SIZE / WARP_THREAD_NUM;
    constexpr size_t ROUND_PER_WARP = 2; // fix for this work
    constexpr size_t WARP_SIZE_BYTE_NUM = sizeof(uint16_t) * WARP_NUM * ROUND_PER_WARP;
    constexpr size_t MIN_BYTE_NUM_PER_ROUND_PER_WARP = PACK_PER_THREAD * RAW_BIT_LEN * WARP_THREAD_NUM / 8;
    constexpr size_t ENCODE_LEN_BYTE_NUM_PER_ROUND_PER_WARP = min_bits_for_encode_len(RAW_BIT_LEN) * PACK_PER_THREAD * WARP_THREAD_NUM / 8;

    uint16_t round_size[WARP_NUM * ROUND_PER_WARP] = {0};

    copy_data((const uint8_t *)encode_data, (uint8_t *)round_size, WARP_SIZE_BYTE_NUM);

    size_t offset = WARP_SIZE_BYTE_NUM;
    for (size_t warp_idx = 0; warp_idx < WARP_NUM; warp_idx++) {
        // std::cout << "round_idx: " << round_idx << std::endl;
        v_bits_warp_uncompact<RAW_BIT_LEN, ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, MIN_BYTE_NUM_PER_ROUND_PER_WARP, PACK_PER_THREAD>(encode_data + offset, mins, 0, warp_idx);
        offset += MIN_BYTE_NUM_PER_ROUND_PER_WARP;
        v_bits_warp_uncompact<min_bits_for_encode_len(RAW_BIT_LEN), ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, ENCODE_LEN_BYTE_NUM_PER_ROUND_PER_WARP, PACK_PER_THREAD>(encode_data + offset, encode_lens, 0, warp_idx);
        offset += ENCODE_LEN_BYTE_NUM_PER_ROUND_PER_WARP;
        size_t in_warp_round_size = v_encoded_packs_uncompact<ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, PACK_PER_THREAD>(encoded_packs, encode_lens, encode_data + offset, 0, warp_idx);
        offset += in_warp_round_size;
        // std::cout << "warp_idx: " << warp_idx << ", in_warp_round_idx: " << 0 << ", encoded_size: " << in_warp_round_size << std::endl;
        v_bits_warp_uncompact<RAW_BIT_LEN, ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, MIN_BYTE_NUM_PER_ROUND_PER_WARP, PACK_PER_THREAD>(encode_data + offset, mins, 1, warp_idx);
        offset += MIN_BYTE_NUM_PER_ROUND_PER_WARP;
        v_bits_warp_uncompact<min_bits_for_encode_len(RAW_BIT_LEN), ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, ENCODE_LEN_BYTE_NUM_PER_ROUND_PER_WARP, PACK_PER_THREAD>(encode_data + offset, encode_lens, 1, warp_idx);
        offset += ENCODE_LEN_BYTE_NUM_PER_ROUND_PER_WARP;
        in_warp_round_size = v_encoded_packs_uncompact<ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, PACK_PER_THREAD>(encoded_packs, encode_lens, encode_data + offset, 1, warp_idx);
        offset += in_warp_round_size;
        // std::cout << "warp_idx: " << warp_idx << ", in_warp_round_idx: " << 1 << ", encoded_size: " << in_warp_round_size << std::endl;
    }

    return;
}

template<size_t RAW_BIT_LEN, size_t HIDDEN_DIM_BLOCK_SIZE, size_t CTX_LEN_BLOCK_SIZE, size_t PACK_PER_THREAD>
void v_block_decode_cpu(
    const uint8_t compressed_buffer[CTX_LEN_BLOCK_SIZE * HIDDEN_DIM_BLOCK_SIZE],
    uint8_t k_encode_block[CTX_LEN_BLOCK_SIZE][HIDDEN_DIM_BLOCK_SIZE]
) {
    constexpr size_t ROUND_NUM = CTX_LEN_BLOCK_SIZE / PACK_ELE_NUM;

    uint8_t mins[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE];
    uint8_t encode_lens[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE];
    uint64_t encoded_packs[ROUND_NUM][HIDDEN_DIM_BLOCK_SIZE];

    v_data_uncompact<RAW_BIT_LEN, ROUND_NUM, CTX_LEN_BLOCK_SIZE, HIDDEN_DIM_BLOCK_SIZE, PACK_PER_THREAD>(mins, encode_lens, encoded_packs, compressed_buffer);

    decode_packs<ROUND_NUM, HIDDEN_DIM_BLOCK_SIZE, CTX_LEN_BLOCK_SIZE, PACK_PER_THREAD>(k_encode_block, encoded_packs, mins, encode_lens);

    // std::cout << "mins: " << std::endl;
    // for (size_t round_idx = 0; round_idx < ROUND_NUM; round_idx++) {
    //     for (size_t in_round_pack_idx = 0; in_round_pack_idx < HIDDEN_DIM_BLOCK_SIZE; in_round_pack_idx ++) {
    //         std::cout << (int)mins[round_idx][in_round_pack_idx] << ", ";
    //     }
    //     std::cout << std::endl;
    // }
    //
    // std::cout << "encode_lens: " << std::endl;
    // for (size_t round_idx = 0; round_idx < ROUND_NUM; round_idx++) {
    //     for (size_t in_round_pack_idx = 0; in_round_pack_idx < HIDDEN_DIM_BLOCK_SIZE; in_round_pack_idx ++) {
    //         std::cout << (int)encode_lens[round_idx][in_round_pack_idx] << ", ";
    //     }
    //     std::cout << std::endl;
    // }

    return;
}

template<size_t RAW_BIT_LEN, size_t HIDDEN_DIM, size_t HIDDEN_DIM_BLOCK_SIZE, size_t CTX_LEN_BLOCK_SIZE>
void v_decode_cpu(
    const uint8_t *compressed_buffer_ptr,
    const uint32_t *block_info_buffer_ptr,
    uint8_t *quant_ints_tensor_ptr,
    size_t ctx_len
) {
    uint8_t compressed_buffer[CTX_LEN_BLOCK_SIZE * HIDDEN_DIM_BLOCK_SIZE] = {0};
    uint8_t k_encode_block[CTX_LEN_BLOCK_SIZE][HIDDEN_DIM_BLOCK_SIZE] = {0};

    size_t block_num = ctx_len / CTX_LEN_BLOCK_SIZE * HIDDEN_DIM / HIDDEN_DIM_BLOCK_SIZE;
    size_t hidden_dim_block_num = HIDDEN_DIM / HIDDEN_DIM_BLOCK_SIZE;

    for (size_t block_idx = 0; block_idx < block_num; block_idx++) {
        size_t block_compressed_size = block_info_buffer_ptr[block_idx * 2 + 1];
        size_t block_compressed_offset = block_info_buffer_ptr[block_idx * 2];
        copy_data(compressed_buffer_ptr + block_compressed_offset, compressed_buffer, block_compressed_size);
        v_block_decode_cpu<RAW_BIT_LEN, HIDDEN_DIM_BLOCK_SIZE, CTX_LEN_BLOCK_SIZE, 4>(compressed_buffer, k_encode_block);
        size_t ctx_len_block_idx = block_idx / hidden_dim_block_num;
        size_t hidden_dim_block_idx = block_idx % hidden_dim_block_num;

        // for (size_t i = 0; i < CTX_LEN_BLOCK_SIZE; i++) {
        //     for (size_t j = 0; j < HIDDEN_DIM_BLOCK_SIZE; j++) {
        //         std::cout << (int)k_encode_block[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;

        store_2d_block<HIDDEN_DIM, HIDDEN_DIM_BLOCK_SIZE, CTX_LEN_BLOCK_SIZE>(quant_ints_tensor_ptr, k_encode_block, ctx_len_block_idx, hidden_dim_block_idx);
    }

    return;
}

}

void v_decode_cpu_assert(
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

void v_decode_cpu_pyi(
    const torch::Tensor &compressed_buffer,
    const torch::Tensor &block_info_buffer,
    torch::Tensor &quant_ints_tensor,
    const int ctx_len,
    const int hidden_dim,
    const int ctx_len_block_size,
    const int hidden_dim_block_size,
    const int bits_len
) {
    v_decode_cpu_assert(quant_ints_tensor, compressed_buffer, block_info_buffer, ctx_len, hidden_dim, ctx_len_block_size, hidden_dim_block_size, bits_len);

    uint8_t *quant_ints_tensor_ptr = quant_ints_tensor.data_ptr<uint8_t>();
    const uint8_t *compressed_buffer_ptr = compressed_buffer.data_ptr<uint8_t>();
    const uint32_t *block_info_buffer_ptr = block_info_buffer.data_ptr<uint32_t>();
    const int ctx_len_block_num = ctx_len / ctx_len_block_size;
    const int hidden_dim_block_num = hidden_dim / hidden_dim_block_size;

    if (
        ctx_len_block_size == 128 &&
        hidden_dim_block_size == 64 &&
        hidden_dim == 5120 &&
        bits_len == 3
    ){
        v_cpu::v_decode_cpu<3, 5120, 64, 128>(
            compressed_buffer_ptr,
            block_info_buffer_ptr,
            quant_ints_tensor_ptr,
            ctx_len
        );
    } else if (
        ctx_len_block_size == 128 &&
        hidden_dim_block_size == 64 &&
        hidden_dim == 1024 &&
        bits_len == 3
    ){
        v_cpu::v_decode_cpu<3, 1024, 64, 128>(
            compressed_buffer_ptr,
            block_info_buffer_ptr,
            quant_ints_tensor_ptr,
            ctx_len
        );
    } else {
        throw std::runtime_error("not supported");
    }
}