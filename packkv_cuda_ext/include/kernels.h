#ifndef PACKKV_KERNELS_H
#define PACKKV_KERNELS_H

#include <torch/torch.h>
#include <utility>
#include <static_def.h>

float kq_mat_vec_mul(
        const torch::Tensor &compressed_buffer,
        const torch::Tensor &block_info_buffer,
        const torch::Tensor &q,
        torch::Tensor &kq_out,
        const int ctx_len,
        const int hidden_dim,
        const int ctx_len_block_size,
        const int hidden_dim_block_size,
        const int bits_len
);

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
);


#endif // PACKKV_KERNELS_H