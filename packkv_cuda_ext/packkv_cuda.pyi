from typing import Tuple
import torch

def k_encode_cpu(
        quant_ints_tensor: torch.Tensor,
        compressed_buffer: torch.Tensor,
        block_info_buffer: torch.Tensor,
        ctx_len: int,
        hidden_dim: int,
        ctx_len_block_size: int,
        hidden_dim_block_size: int,
        bits_len: int
) -> int: ...

def k_decode_cpu(
        compressed_buffer: torch.Tensor,
        block_info_buffer: torch.Tensor,
        quant_ints_tensor: torch.Tensor,
        ctx_len: int,
        hidden_dim: int,
        ctx_len_block_size: int,
        hidden_dim_block_size: int,
        bits_len: int
) -> None: ...

def kq_mat_vec_mul(
    compressed_buffer: torch.Tensor,
    block_info_buffer: torch.Tensor,
    q: torch.Tensor,
    kq_out: torch.Tensor,
    ctx_len: int,
    hidden_dim: int,
    ctx_len_block_size: int,
    hidden_dim_block_size: int,
    bits_len: int
) -> float: ...

def v_encode_cpu(
        quant_ints_tensor: torch.Tensor,
        compressed_buffer: torch.Tensor,
        block_info_buffer: torch.Tensor,
        ctx_len: int,
        hidden_dim: int,
        ctx_len_block_size: int,
        hidden_dim_block_size: int,
        bits_len: int
) -> int: ...

def v_decode_cpu(
        compressed_buffer: torch.Tensor,
        block_info_buffer: torch.Tensor,
        quant_ints_tensor: torch.Tensor,
        ctx_len: int,
        hidden_dim: int,
        ctx_len_block_size: int,
        hidden_dim_block_size: int,
        bits_len: int
) -> None: ...

def wv_mat_vec_mul(
    compressed_buffer: torch.Tensor,
    block_info_buffer: torch.Tensor,
    w: torch.Tensor,
    wv_out: torch.Tensor,
    ctx_len: int,
    hidden_dim: int,
    ctx_len_block_size: int,
    hidden_dim_block_size: int,
    bits_len: int
) -> float: ...

def fused_kq(
    our_kq: torch.Tensor,
    q: torch.Tensor,
    k_quant_zero: torch.Tensor,
    k_quant_scale: torch.Tensor,
    t: torch.Tensor
) -> float: ...

def fused_wv(
    our_wv: torch.Tensor,
    w: torch.Tensor,
    v_quant_zero: torch.Tensor,
    v_quant_scale: torch.Tensor,
    term1: torch.Tensor
) -> float: ...
