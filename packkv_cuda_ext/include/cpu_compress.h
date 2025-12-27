#ifndef PACKKV_CPU_COMPRESS_H
#define PACKKV_CPU_COMPRESS_H

#include <torch/torch.h>
#include <utility>
#include <static_def.h>

size_t k_encode_cpu_pyi(
    const torch::Tensor &quant_ints_tensor,
    torch::Tensor &compressed_buffer,
    torch::Tensor &block_info_buffer,
    const int ctx_len,
    const int hidden_dim,
    const int ctx_len_block_size,
    const int hidden_dim_block_size,
    const int bits_len
);

void k_decode_cpu_pyi(
    const torch::Tensor &compressed_buffer,
    const torch::Tensor &block_info_buffer,
    torch::Tensor &quant_ints_tensor,
    const int ctx_len,
    const int hidden_dim,
    const int ctx_len_block_size,
    const int hidden_dim_block_size,
    const int bits_len
);

size_t v_encode_cpu_pyi(
    const torch::Tensor &quant_ints_tensor,
    torch::Tensor &compressed_buffer,
    torch::Tensor &block_info_buffer,
    const int ctx_len,
    const int hidden_dim,
    const int ctx_len_block_size,
    const int hidden_dim_block_size,
    const int bits_len
);

void v_decode_cpu_pyi(
    const torch::Tensor &compressed_buffer,
    const torch::Tensor &block_info_buffer,
    torch::Tensor &quant_ints_tensor,
    const int ctx_len,
    const int hidden_dim,
    const int ctx_len_block_size,
    const int hidden_dim_block_size,
    const int bits_len
);

#endif //PACKKV_CPU_COMPRESS_H
