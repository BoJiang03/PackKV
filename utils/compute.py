import torch
from typing import Tuple, Optional, List
from enum import Enum
import math

class QuantMode(Enum):
    # LayerQuant = "LayerQuant"
    BlockQuant = "BlockQuant"
    ChannelQuant = "ChannelQuant"
    TokenQuant = "TokenQuant"
    VectorQuant = "VectorQuant"

class QuantMethod(Enum):
    KIVI = (QuantMode.ChannelQuant, QuantMode.TokenQuant)
    PackKV = (QuantMode.TokenQuant, QuantMode.TokenQuant)

class RepackMethod(Enum):
    GREEDY = "Greedy"
    MEDIAN = "Median"
    NONE = "None"

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_single(
        t: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids=None, unsqueeze_dim=1) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    t_embed = (t * cos) + (rotate_half(t) * sin)
    return t_embed

def safe_cat(t1, t2, dim):
    if t1 is None and t2 is None:
        return None
    if t1 is None:
        return t2.clone()
    if t2 is None:
        return t1.clone()
    return torch.cat([t1, t2], dim=dim)

def cut_tensor(
        buffer, new_tensor,
        block_size, recent_size,
        dim=2
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    buffer = safe_cat(buffer, new_tensor, dim)
    len_ = buffer.shape[dim]
    res_num = len_ % block_size
    to_compress_block_num = (len_ + block_size - res_num - recent_size) // block_size
    to_compress = None
    if to_compress_block_num > 0:
        to_compress = buffer[:, :, :to_compress_block_num * block_size, :]
        buffer = buffer[:, :, to_compress_block_num * block_size:, :]
    return to_compress, buffer

def cut_tensor_ctx_len_0(
        buffer, new_tensor,
        block_size, recent_size,
        dim=2
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    buffer = safe_cat(buffer, new_tensor, dim)
    len_ = buffer.shape[dim]
    res_num = len_ % block_size
    to_compress_block_num = (len_ + block_size - res_num - recent_size) // block_size
    to_compress = None
    if to_compress_block_num > 0:
        to_compress = buffer[:to_compress_block_num * block_size, :, :, :]
        buffer = buffer[to_compress_block_num * block_size:, :, :, :]
    return to_compress, buffer

def quant_ints(
        tensor: torch.Tensor,
        block_size: int,
        quant_scale_rel: float,
        quant_mode: QuantMode,
        high_precision_zero_point: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert tensor.shape[2] % block_size == 0, "Tensor shape is not divisible by block size"
    tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1, block_size, tensor.shape[3])
    quant_dim = QUANT_DIM[quant_mode.value]

    min_val = tensor
    max_val = tensor
    for i in quant_dim:
        min_val = min_val.min(dim=i, keepdim=True).values
        max_val = max_val.max(dim=i, keepdim=True).values

    if high_precision_zero_point:
        # -min_val, /scale
        quant_scale = (max_val - min_val) * quant_scale_rel
        value_quant = ((tensor - min_val)/ quant_scale).round()
    else:
        # /scale, -min_int
        quant_scale = (max_val - min_val) * quant_scale_rel
        min_int = (min_val / quant_scale).round()
        value_quant = (tensor / quant_scale).round() - min_int
        min_val = min_int

    return value_quant, min_val, quant_scale

def quant_ints_throughput(
        tensor: torch.Tensor,
        block_size: int,
        quant_scale_rel: float,
        quant_mode: QuantMode,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert tensor.shape[2] % block_size == 0, "Tensor shape is not divisible by block size"
    tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1, block_size, tensor.shape[3])
    quant_dim = QUANT_DIM[quant_mode.value]

    min_val = torch.amin(tensor, dim=quant_dim, keepdim=True)
    max_val = torch.amax(tensor, dim=quant_dim, keepdim=True)

    quant_scale = (max_val - min_val) * quant_scale_rel
    min_int = (min_val / quant_scale).round()
    value_quant = (tensor / quant_scale).round() - min_int
    min_val = min_int

    return value_quant, min_val, quant_scale

def quant(
        tensor: torch.Tensor,
        quant_dims: List[int],
        quant_scale_rel: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_ = tensor
    max_ = tensor
    for dim in quant_dims:
        min_ = min_.min(dim=dim, keepdim=True).values
        max_ = max_.max(dim=dim, keepdim=True).values
    quant_scale = (max_ - min_) * quant_scale_rel
    min_ints = (min_ / quant_scale).round_() #.to(torch.int8)
    quant_ints = (tensor / quant_scale).round_() #.to(torch.int8)
    return quant_ints - min_ints, min_ints, quant_scale

def quant_error(
        error_cache: torch.Tensor,
        buffer: torch.Tensor,
        new_tensor: torch.Tensor,
        block_size: int,
        recent_size: int,
        quant_scale_rel: float,
        quant_mode: QuantMode,
        high_precision_zero_point: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    to_compress, in_buffer = cut_tensor(
        buffer, new_tensor,
        block_size, recent_size,
        dim=2
    )

    if to_compress is not None:
        quant_int, quant_zero, quant_scale = quant_ints(
            to_compress,
            block_size,
            quant_scale_rel,
            quant_mode,
            high_precision_zero_point
        )
        if high_precision_zero_point:
            # -min_val, /scale
            to_compress = quant_int * quant_scale + quant_zero
        else:
            # /scale, -min_int
            to_compress = (quant_int + quant_zero)* quant_scale
        to_compress = to_compress.reshape(to_compress.shape[0], to_compress.shape[1], -1, to_compress.shape[4])

    return safe_cat(error_cache, to_compress, dim=2), in_buffer

def print_quant_setting(logger):
    logger.info(QUANT_DIM)

def _batched_pick(tensor: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Picks a vector from each batch element and returns the picked vectors and the remaining tensors.
    tensor: (B, N, D)
    indices: (B,)
    Returns: (picked_vectors (B, D), remaining_tensor (B, N-1, D))
    """
    B, N, D = tensor.shape
    device = tensor.device

    idx_expanded = indices.view(B, 1, 1).expand(-1, 1, D)
    picked_vectors = torch.gather(tensor, 1, idx_expanded).squeeze(1)

    if N == 1:
        remained_tensor = torch.empty((B, 0, D), dtype=tensor.dtype, device=device)
        return picked_vectors, remained_tensor

    mask = torch.ones(B, N, device=device, dtype=torch.bool)
    batch_indices = torch.arange(B, device=device)
    mask[batch_indices, indices] = False
    remained_tensor = tensor[mask].view(B, N - 1, D)

    return picked_vectors, remained_tensor

def greedy_repacking(blocks: torch.Tensor, pack_len: int) -> torch.Tensor:
    B, N, D = blocks.shape

    remaining_blocks = blocks.clone().to(torch.float32)

    pack_num = N // pack_len

    repacked_packs_list = []

    for _ in range(pack_num):
        mean_vectors = remaining_blocks.mean(dim=1, keepdim=True).round()
        cosine_sim = torch.nn.functional.cosine_similarity(remaining_blocks, mean_vectors, dim=2, eps=1e-8)
        max_sim_indices = torch.argmax(cosine_sim, dim=1)

        seed_vectors, remaining_blocks = _batched_pick(remaining_blocks, max_sim_indices)

        pack_tensor_list = [seed_vectors.unsqueeze(1)]

        mins_ = seed_vectors
        maxs_ = seed_vectors

        for _ in range(pack_len - 1):
            if remaining_blocks.shape[1] == 0:
                break

            current_mins = mins_.unsqueeze(1)
            current_maxs = maxs_.unsqueeze(1)

            pre_status_num = torch.ceil(torch.log2(current_maxs - current_mins + 1))

            all_possible_max = torch.max(remaining_blocks, current_maxs)
            all_possible_min = torch.min(remaining_blocks, current_mins)

            all_possible_bit_num = torch.ceil(torch.log2(all_possible_max - all_possible_min + 1))
            all_possible_bit_num_increase = (all_possible_bit_num - pre_status_num).sum(dim=2)

            selected_vector_indices = torch.argmin(all_possible_bit_num_increase, dim=1)

            selected_vectors, remaining_blocks = _batched_pick(remaining_blocks, selected_vector_indices)

            pack_tensor_list.append(selected_vectors.unsqueeze(1))

            mins_ = torch.min(mins_, selected_vectors)
            maxs_ = torch.max(maxs_, selected_vectors)

        pack = torch.cat(pack_tensor_list, dim=1)
        repacked_packs_list.append(pack)

    repacked_blocks = torch.cat(repacked_packs_list, dim=1)

    return repacked_blocks.to(torch.int32)

def median_repacking(blocks: torch.Tensor) -> torch.Tensor:
    B, N, D = blocks.shape
    half_vec_len = D // 2
    v_part = blocks[:, :, half_vec_len:]
    median_values = torch.median(v_part, dim=2).values
    _, sorted_indices = torch.sort(median_values, dim=1, descending=True)
    sorted_indices_expanded = sorted_indices.unsqueeze(2).expand(B, N, D)
    repacked_blocks = torch.gather(blocks, 1, sorted_indices_expanded)
    return repacked_blocks

def bit_pack(
        blocks: torch.Tensor,
        pack_len: int
)-> Tuple[int, int]:
    half_vec_len = blocks.shape[2] // 2
    blocks = blocks.flatten(0, 1).to(torch.int64)
    k_blocks = blocks[:, :half_vec_len]
    v_blocks = blocks[:, half_vec_len:]

    k_packs = k_blocks.view(-1, pack_len, half_vec_len)
    v_packs = v_blocks.view(-1, pack_len, half_vec_len)

    k_bit_len = math.ceil(math.log2(k_packs.unique().numel()))
    v_bit_len = math.ceil(math.log2(v_packs.unique().numel()))

    k_pack_mins = k_packs.min(dim=1).values
    k_pack_maxs = k_packs.max(dim=1).values
    v_pack_mins = v_packs.min(dim=1).values
    v_pack_maxs = v_packs.max(dim=1).values

    k_pack_bit_num = torch.ceil(torch.log2(k_pack_maxs - k_pack_mins + 1)).to(torch.int64).sum() * pack_len
    v_pack_bit_num = torch.ceil(torch.log2(v_pack_maxs - v_pack_mins + 1)).to(torch.int64).sum() * pack_len
    k_pack_bit_num += k_pack_mins.numel() * (k_bit_len + math.ceil(math.log2(k_bit_len + 1)))
    v_pack_bit_num += v_pack_mins.numel() * (v_bit_len + math.ceil(math.log2(v_bit_len + 1)))

    return k_pack_bit_num.item() // 8, v_pack_bit_num.item() // 8

def bit_pack_detail_rebuttal(
        blocks: torch.Tensor,
        pack_len: int
)-> Tuple[int, int, int, int, int, int]:
    half_vec_len = blocks.shape[2] // 2
    blocks = blocks.flatten(0, 1).to(torch.int64)
    k_blocks = blocks[:, :half_vec_len]
    v_blocks = blocks[:, half_vec_len:]

    k_packs = k_blocks.view(-1, pack_len, half_vec_len)
    v_packs = v_blocks.view(-1, pack_len, half_vec_len)

    k_bit_len = math.ceil(math.log2(k_packs.unique().numel()))
    v_bit_len = math.ceil(math.log2(v_packs.unique().numel()))

    k_pack_mins = k_packs.min(dim=1).values
    k_pack_maxs = k_packs.max(dim=1).values
    v_pack_mins = v_packs.min(dim=1).values
    v_pack_maxs = v_packs.max(dim=1).values

    k_pack_bit_num = torch.ceil(torch.log2(k_pack_maxs - k_pack_mins + 1)).to(torch.int64).sum() * pack_len
    v_pack_bit_num = torch.ceil(torch.log2(v_pack_maxs - v_pack_mins + 1)).to(torch.int64).sum() * pack_len
    k_zero_point_bit_num = k_pack_mins.numel() * k_bit_len 
    k_encode_len_bit_num = k_pack_mins.numel() * math.ceil(math.log2(k_bit_len + 1))
    v_zero_point_bit_num = v_pack_mins.numel() * v_bit_len 
    v_encode_len_bit_num = v_pack_mins.numel() * math.ceil(math.log2(v_bit_len + 1))

    return k_zero_point_bit_num // 8, v_zero_point_bit_num // 8, k_encode_len_bit_num // 8, v_encode_len_bit_num // 8, k_pack_bit_num.item() // 8, v_pack_bit_num.item() // 8


def repack_and_encode(
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    pack_size: int,
    repack_method: RepackMethod,
    before_and_after_repacking = None
)-> Tuple[int, int, int, int]:
    k_blocks = k_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    v_blocks = v_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    blocks = torch.cat([k_blocks, v_blocks], dim=2)
    k_size_pre, v_size_pre = bit_pack(blocks, pack_size)
    before_and_after_ = [blocks, None]
    if repack_method == RepackMethod.GREEDY:
        blocks = greedy_repacking(blocks, pack_size)
    elif repack_method == RepackMethod.MEDIAN:
        blocks = median_repacking(blocks)
    elif repack_method == RepackMethod.NONE:
        pass
    else:
        raise ValueError(f"repack_method must be one of {RepackMethod.__members__.keys()}")

    before_and_after_[1] = blocks

    if before_and_after_repacking is not None:
        before_and_after_repacking.append(before_and_after_)

    k_size_aft, v_size_aft = bit_pack(blocks, pack_size)

    return k_size_pre, v_size_pre, k_size_aft, v_size_aft

def repack_and_encode_detail_rebuttal(
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    pack_size: int,
)-> Tuple[int, int, int, int, int, int]:
    k_blocks = k_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    v_blocks = v_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    blocks = torch.cat([k_blocks, v_blocks], dim=2)
    blocks = median_repacking(blocks)
    k_zero_point_size, v_zero_point_size, k_encode_len_size, v_encode_len_size, k_pack_size, v_pack_size = bit_pack_detail_rebuttal(blocks, pack_size)

    return k_zero_point_size, v_zero_point_size, k_encode_len_size, v_encode_len_size, k_pack_size, v_pack_size

def repack_throughput_detail_rebuttal(
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    pack_size: int,
)-> Tuple[float, float]:
    k_blocks = k_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    v_blocks = v_tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)
    blocks_ = torch.cat([k_blocks, v_blocks], dim=2)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    blocks = greedy_repacking(blocks_, pack_size)
    end.record()
    torch.cuda.synchronize()
    greedy_time = start.elapsed_time(end)

    start.record()
    blocks = median_repacking(blocks_)
    end.record()
    torch.cuda.synchronize()
    median_time = start.elapsed_time(end)

    return greedy_time, median_time


def entropy(tensor):
    values, counts = torch.unique(tensor, return_counts=True)
    probs = counts.float() / counts.sum()
    entropy = -torch.sum(probs * torch.log2(probs))
    return entropy

QUANT_DIM = {
    QuantMode.BlockQuant.value: [1,3,4],
    QuantMode.ChannelQuant.value: [3],
    QuantMode.TokenQuant.value: [1,4],
    QuantMode.VectorQuant.value: [4],
}

