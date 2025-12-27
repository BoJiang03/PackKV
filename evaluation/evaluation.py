from typing import Tuple, List

import torch
import os
import math

from torch import nn

from models.cache.packkv_quant import PackKVCacheConfigStatic, PackKVCachePytorchQuant
from models.llama import LlamaForCausalLM
from models.phi import Phi3ForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
from lm_eval import evaluator

from models.qwen3 import Qwen3ForCausalLM
from models.mistral import MistralForCausalLM
from utils.compute import QuantMode, QuantMethod, quant_ints, repack_and_encode, repack_and_encode_detail_rebuttal, \
    repack_throughput_detail_rebuttal, quant_ints_throughput
from utils.config import PackKVCacheConfig, ExtractCacheConfig
from utils.lm_eval_warp import LMEvalWrapper
from utils.profiling_func import profile_func
from utils.util import JumpOutException
from packkv_cuda import k_encode_cpu, k_decode_cpu, kq_mat_vec_mul, v_encode_cpu, v_decode_cpu, wv_mat_vec_mul, \
    fused_kq, fused_wv

from datasets import load_dataset

MODEL_CLASS_MAP = {
    "meta-llama/Llama-2-7b-hf": LlamaForCausalLM,
    "meta-llama/Llama-2-13b-hf": LlamaForCausalLM,
    "meta-llama/Llama-3.1-8B": LlamaForCausalLM,
    "meta-llama/Meta-Llama-3-8B": LlamaForCausalLM,
    "Qwen/Qwen3-8B": Qwen3ForCausalLM,
    "Qwen/Qwen3-4B": Qwen3ForCausalLM,
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": Qwen3ForCausalLM,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": LlamaForCausalLM,
    "mistralai/Ministral-8B-Instruct-2410": MistralForCausalLM,
    "microsoft/phi-4": Phi3ForCausalLM
}


def accuracy_evaluation(
        config: PackKVCacheConfig,
        benchmark: str | list[str],
        logger
):
    logger.info(f"\n{benchmark}:\n{config}")
    PackKVCacheConfigStatic.config = config
    tokenizer = AutoTokenizer.from_pretrained(PackKVCacheConfigStatic.config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        PackKVCacheConfigStatic.config.model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    logger.info(f"model class: {model_class.__name__}")
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    batch_size = 4

    lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)

    results = evaluator.simple_evaluate(
        model=lm_eval_warp,
        tasks=[benchmark] if isinstance(benchmark, str) else benchmark,
        # num_fewshot=0,  # Number of few-shot examples
        batch_size=batch_size,
        # device=_device,
    )

    PackKVCacheConfigStatic.config = None
    return results['results']


def accuracy_evaluation_with_model(
        model,
        tokenizer,
        benchmark: str | list[str],
):
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    batch_size = 1

    lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)

    results = evaluator.simple_evaluate(
        model=lm_eval_warp,
        tasks=[benchmark] if isinstance(benchmark, str) else benchmark,
        # num_fewshot=0,  # Number of few-shot examples
        batch_size=batch_size,
        # device=_device,
    )

    PackKVCacheConfigStatic.config = None
    return results['results']


def get_collected_data(
        logger,
        model_name="meta-llama/Llama-2-13b-hf"
):
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,
        model_name=model_name,
        quant_method=QuantMethod.PackKV,
        k_block_size=64,
        v_block_size=128,
        k_buffer_size=128,
        v_buffer_size=256,
        k_quant_scale_rel=0.01,
        v_quant_scale_rel=0.01
    )
    PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_class = MODEL_CLASS_MAP.get(model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {model_name}")
    model = model_class.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    batch_size = 1
    lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    try:
        evaluator.simple_evaluate(
            model=lm_eval_warp,
            tasks=["gsm8k"],
            batch_size=batch_size,
        )
    except JumpOutException as e:
        logger.info(e.message)

    rt = PackKVCacheConfigStatic.extract_cache
    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None

    return rt


def get_ctx_len_text_from_wikitext_103_v1(
        ctx_len: int,
        tokenizer
):
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    rt_text = ""
    for i in range(len(dataset['train'])):
        text = dataset['train'][i]['text']
        rt_text += text
        tokens = tokenizer(rt_text, return_tensors="pt")['input_ids']
        if tokens.shape[1] > ctx_len:
            break
    return tokenizer(rt_text, return_tensors="pt", truncation=True, max_length=ctx_len)


def save_extract_cache(
        model_name: str,
        ctx_len: int,
        extract_cache: ExtractCacheConfig,
        root_dir: str
):
    # check whether the directory exists
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for round_i in range(extract_cache.collect_round):
        save_dir = os.path.join(root_dir, model_name, str(ctx_len), str(round_i))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        k_dir = os.path.join(save_dir, "k")
        v_dir = os.path.join(save_dir, "v")
        if not os.path.exists(k_dir):
            os.makedirs(k_dir)
        if not os.path.exists(v_dir):
            os.makedirs(v_dir)

        key_caches = PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values()
        value_caches = PackKVCacheConfigStatic.extract_cache.value_caches[round_i].values()
        for i, key_cache in enumerate(key_caches):
            torch.save(key_cache, os.path.join(k_dir, f"{i}.pt"))
        for i, value_cache in enumerate(value_caches):
            torch.save(value_cache, os.path.join(v_dir, f"{i}.pt"))


def crs_evaluation_with_data(
        config: PackKVCacheConfig,
        key_caches,
        value_caches,
        before_and_after_repacking=None
):
    # assert key_caches[0].shape[1] == 32 and key_caches[0].shape[3] == 128
    res = {
        "k_original_size": [],
        "v_original_size": [],
        "k_quant_size": [],
        "v_quant_size": [],
        "k_encode_size_before_repack": [],
        "v_encode_size_before_repack": [],
        "k_encode_size_after_repack": [],
        "v_encode_size_after_repack": [],
        "k_quant_cr": [],
        "v_quant_cr": [],
        "k_encode_before_repack_cr": [],
        "v_encode_before_repack_cr": [],
        "k_encode_after_repack_cr": [],
        "v_encode_after_repack_cr": []
    }

    layer_num = len(key_caches)

    for layer_idx in range(layer_num):
        k = key_caches[layer_idx]
        v = value_caches[layer_idx]
        k_origin_size = k.numel() * k.element_size()
        v_origin_size = v.numel() * v.element_size()
        res["k_original_size"].append(k_origin_size)
        res["v_original_size"].append(v_origin_size)
        k_quant_int, k_quant_zero, k_quant_scale = quant_ints(k, config.block_size, config.k_quant_scale_rel,
                                                              config.quant_method.value[0])
        v_quant_int, v_quant_zero, v_quant_scale = quant_ints(v, config.block_size, config.v_quant_scale_rel,
                                                              config.quant_method.value[1])
        # k_quant_int = k_quant_int.flatten(2,3)
        # v_quant_int = v_quant_int.flatten(2,3)
        k_quant_bit_num = math.ceil(math.log2(k_quant_int.unique().numel()))
        v_quant_bit_num = math.ceil(math.log2(v_quant_int.unique().numel()))
        k_quant_size = k_quant_int.numel() * k_quant_bit_num // 8 + k_quant_zero.numel() * k_quant_zero.element_size() + k_quant_scale.numel() * k_quant_scale.element_size()
        v_quant_size = v_quant_int.numel() * v_quant_bit_num // 8 + v_quant_zero.numel() * v_quant_zero.element_size() + v_quant_scale.numel() * v_quant_scale.element_size()
        res["k_quant_size"].append(k_quant_size)
        res["v_quant_size"].append(v_quant_size)
        k_quant_cr = k_origin_size / k_quant_size
        v_quant_cr = v_origin_size / v_quant_size
        res["k_quant_cr"].append(k_quant_cr)
        res["v_quant_cr"].append(v_quant_cr)
        if config.quant_method == QuantMethod.PackKV:
            k_encode_size_before_repack, v_encode_size_before_repack, k_encode_size_after_repack, v_encode_size_after_repack = repack_and_encode(
                k_quant_int, v_quant_int, config.pack_size, config.repack_method, before_and_after_repacking)
            res["k_encode_size_before_repack"].append(k_encode_size_before_repack)
            res["v_encode_size_before_repack"].append(v_encode_size_before_repack)
            res["k_encode_size_after_repack"].append(k_encode_size_after_repack)
            res["v_encode_size_after_repack"].append(v_encode_size_after_repack)
            k_encode_before_repack_cr = k_origin_size / k_encode_size_before_repack
            v_encode_before_repack_cr = v_origin_size / v_encode_size_before_repack
            res["k_encode_before_repack_cr"].append(k_encode_before_repack_cr)
            res["v_encode_before_repack_cr"].append(v_encode_before_repack_cr)
            k_encode_after_repack_cr = k_origin_size / k_encode_size_after_repack
            v_encode_after_repack_cr = v_origin_size / v_encode_size_after_repack
            res["k_encode_after_repack_cr"].append(k_encode_after_repack_cr)
            res["v_encode_after_repack_cr"].append(v_encode_after_repack_cr)
    return res


def crs_evaluation_with_data_detail_rebuttal(
        config: PackKVCacheConfig,
        key_caches,
        value_caches,
        before_and_after_repacking=None
):
    # assert key_caches[0].shape[1] == 32 and key_caches[0].shape[3] == 128
    res = {
        "k_original_size": [],
        "v_original_size": [],
        "k_quant_zero_point_size": [],
        "v_quant_zero_point_size": [],
        "k_quant_scale_size": [],
        "v_quant_scale_size": [],
        "k_bitpack_min_value_size": [],
        "v_bitpack_min_value_size": [],
        "k_bitpack_encode_len_size": [],
        "v_bitpack_encode_len_size": [],
        "k_bitpack_encoded_size": [],
        "v_bitpack_encoded_size": [],
    }

    layer_num = len(key_caches)

    for layer_idx in range(layer_num):
        k = key_caches[layer_idx]
        v = value_caches[layer_idx]
        k_origin_size = k.numel() * k.element_size()
        v_origin_size = v.numel() * v.element_size()
        res["k_original_size"].append(k_origin_size)
        res["v_original_size"].append(v_origin_size)
        k_quant_int, k_quant_zero, k_quant_scale = quant_ints(k, config.block_size, config.k_quant_scale_rel,
                                                              config.quant_method.value[0])
        v_quant_int, v_quant_zero, v_quant_scale = quant_ints(v, config.block_size, config.v_quant_scale_rel,
                                                              config.quant_method.value[1])
        # k_quant_int = k_quant_int.flatten(2,3)
        # v_quant_int = v_quant_int.flatten(2,3)
        k_quant_bit_num = math.ceil(math.log2(k_quant_int.unique().numel()))
        v_quant_bit_num = math.ceil(math.log2(v_quant_int.unique().numel()))
        res["k_quant_zero_point_size"].append(k_quant_zero.numel() * k_quant_zero.element_size())
        res["v_quant_zero_point_size"].append(v_quant_zero.numel() * v_quant_zero.element_size())
        res["k_quant_scale_size"].append(k_quant_scale.numel() * k_quant_scale.element_size())
        res["v_quant_scale_size"].append(v_quant_scale.numel() * v_quant_scale.element_size())
        if config.quant_method == QuantMethod.PackKV:
            k_bitpack_min_value_size, v_bitpack_min_value_size, k_bitpack_encode_len_size, v_bitpack_encode_len_size, k_bitpack_encoded_size, v_bitpack_encoded_size = repack_and_encode_detail_rebuttal(
                k_quant_int, v_quant_int, config.pack_size)
            res["k_bitpack_min_value_size"].append(k_bitpack_min_value_size)
            res["v_bitpack_min_value_size"].append(v_bitpack_min_value_size)
            res["k_bitpack_encode_len_size"].append(k_bitpack_encode_len_size)
            res["v_bitpack_encode_len_size"].append(v_bitpack_encode_len_size)
            res["k_bitpack_encoded_size"].append(k_bitpack_encoded_size)
            res["v_bitpack_encoded_size"].append(v_bitpack_encoded_size)
    return res


def repacking_throughput_with_kv_rebuttal(
        config: PackKVCacheConfig,
        key_caches,
        value_caches,
):
    # assert key_caches[0].shape[1] == 32 and key_caches[0].shape[3] == 128
    res = {
        "greedy_repacking_time": [],
        "median_repacking_time": []
    }

    layer_num = len(key_caches)

    for layer_idx in range(layer_num):
        k = key_caches[layer_idx]
        v = value_caches[layer_idx]
        k_quant_int, k_quant_zero, k_quant_scale = quant_ints(k, config.block_size, config.k_quant_scale_rel,
                                                              config.quant_method.value[0])
        v_quant_int, v_quant_zero, v_quant_scale = quant_ints(v, config.block_size, config.v_quant_scale_rel,
                                                              config.quant_method.value[1])
        if config.quant_method == QuantMethod.PackKV:
            greedy_time, median_time = repack_throughput_detail_rebuttal(k_quant_int, v_quant_int, config.pack_size)
            res["greedy_repacking_time"].append(greedy_time)
            res["median_repacking_time"].append(median_time)

    return res


def cr_evaluation(
        config: PackKVCacheConfig,
        ctx_len: int,
        enable_save: bool,
        logger,
        collect_round: int = 1,
        before_and_after_repacking=None
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    # logger.info(f"Created a temp config for cache collection and set its enable_quant to False.")
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,
        model_name=config.model_name,
        quant_method=config.quant_method,
        repack_method=config.repack_method,
        high_precision_zero_point=config.high_precision_zero_point,
        block_size=config.block_size,
        buffer_size=config.buffer_size,
        pack_size=config.pack_size,
        k_quant_scale_rel=config.k_quant_scale_rel,
        v_quant_scale_rel=config.v_quant_scale_rel
    )

    PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        PackKVCacheConfigStatic.config.model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    # batch_size = 1
    # lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    inputs = get_ctx_len_text_from_wikitext_103_v1(ctx_len, tokenizer).to(model.device)

    with torch.no_grad():
        _ = model(**inputs)
    PackKVCachePytorchQuant.round_ = 0

    if enable_save:
        save_extract_cache(config.model_name, ctx_len, PackKVCacheConfigStatic.extract_cache, "./dumped_cache")

    rts = []

    logger.info(f"Cache Size {PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024 / 1024} GB")

    for round_i in range(collect_round):
        key_caches = list(PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values())
        value_caches = list(PackKVCacheConfigStatic.extract_cache.value_caches[round_i].values())
        rts.append(crs_evaluation_with_data(config, key_caches, value_caches, before_and_after_repacking))

    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None
    return rts


def k_cpu_compress_gpu_mat_vec_mul(
        k: torch.Tensor,
        config
) -> Tuple[int, float, float, float]:
    k_quant_int, k_quant_zero, k_quant_scale = quant_ints(k, config.block_size, config.k_quant_scale_rel,
                                                          config.quant_method.value[0])
    dtype_ = k.dtype
    k = ((k_quant_int + k_quant_zero) * k_quant_scale).flatten(2, 3)
    k_quant_int = k_quant_int.flatten(2, 3).permute(2, 0, 1, 3).flatten(1, 3).contiguous()
    k_quant_zero = k_quant_zero.flatten()
    k_quant_scale = k_quant_scale.flatten()

    ctx_len_block_size = 64
    hidden_dim_block_size = 128

    ctx_len, hidden_dim = k_quant_int.shape
    ctx_len_block_num = ctx_len // ctx_len_block_size
    hidden_dim_block_num = hidden_dim // hidden_dim_block_size
    k_bit_len = math.ceil(math.log2(k_quant_int.unique().numel()))
    assert k_bit_len == 4

    block = k_quant_int.to(torch.uint8).cpu()
    block_info_buffer = torch.zeros(
        ctx_len_block_num * hidden_dim_block_num,
        2,
        dtype=torch.uint32
    ).cpu()

    compressed_buffer = torch.zeros_like(block)

    compressed_size = k_encode_cpu(
        block,
        compressed_buffer,
        block_info_buffer,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        k_bit_len
    )

    # compressed_buffer = compressed_buffer.flatten()[:compressed_size].clone()

    decompress_tensor = torch.zeros_like(
        block
    )

    k_decode_cpu(
        compressed_buffer,
        block_info_buffer,
        decompress_tensor,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        k_bit_len
    )

    assert (block == decompress_tensor).all()

    head_num = hidden_dim // hidden_dim_block_size
    q = torch.randn(
        head_num,
        hidden_dim_block_size,
        dtype=dtype_
    ).to(k.device)

    q_reshaped = q.view(head_num, 128 // 4, 4)
    part1 = q_reshaped[:, :, :2].contiguous().view(head_num, -1)
    part2 = q_reshaped[:, :, 2:].contiguous().view(head_num, -1)
    q_transform = torch.cat([part1, part2], dim=1)

    t = torch.zeros(
        head_num,
        ctx_len,
        dtype=dtype_
    ).to(k.device)

    our_kernel_time = kq_mat_vec_mul(
        compressed_buffer.to(k.device),
        block_info_buffer.to(k.device),
        q_transform,
        t,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        k_bit_len
    )

    # print("q.shape: ", q.shape, "dtype: ", q.dtype, "is_contiguous: ", q.is_contiguous())
    # print("k_quant_zero.shape", k_quant_zero.shape, "dtype: ", k_quant_zero.dtype, "is_contiguous: ", k_quant_zero.is_contiguous())
    # print("k_quant_scale.shape", k_quant_scale.shape, "dtype: ", k_quant_scale.dtype, "is_contiguous: ", k_quant_scale.is_contiguous())
    # print("t.shape", t.shape, "dtype: ", t.dtype, "is_contiguous: ", t.is_contiguous())

    # Replace Pytorch implementation with our fused CUDA kernel
    our_kq = torch.empty_like(t)
    fused_kernel_time = fused_kq(
        our_kq, q, k_quant_zero, k_quant_scale, t
    )
    our_none_kernel_time = fused_kernel_time

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    k_ = k.squeeze(0)
    q_ = q.unsqueeze(2)
    torch.cuda.synchronize()
    start.record()
    target_kq = k_ @ q_
    end.record()
    torch.cuda.synchronize()
    pytorch_kernel_time = start.elapsed_time(end)

    target_kq = target_kq.squeeze()

    max_error_percentage = ((target_kq - our_kq).abs() / target_kq.abs().max()).max() * 100

    # print(f"max_error_percentage: {max_error_percentage}")
    if max_error_percentage > 10:
        print(f"max_error_percentage: {max_error_percentage}")
        exit()

    # print(f"our_kernel_time: {our_kernel_time}, our_none_kernel_time: {our_none_kernel_time}, pytorch_kernel_time: {pytorch_kernel_time}")
    # exit()

    return compressed_size, our_kernel_time, our_none_kernel_time, pytorch_kernel_time


def k_cpu_compress(
        k: torch.Tensor,
        config
) -> Tuple[int, float, float, float]:
    compress_size = k.numel() * k.element_size()
    compress_time = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    k_quant_int, k_quant_zero, k_quant_scale = quant_ints(k, config.block_size, config.k_quant_scale_rel,
                                                          config.quant_method.value[0])
    dtype_ = k.dtype
    end.record()
    torch.cuda.synchronize()
    compress_time += start.elapsed_time(end)
    k = ((k_quant_int + k_quant_zero) * k_quant_scale).flatten(2, 3)
    k_quant_int = k_quant_int.flatten(2, 3).permute(2, 0, 1, 3).flatten(1, 3).contiguous()
    k_quant_zero = k_quant_zero.flatten()
    k_quant_scale = k_quant_scale.flatten()

    ctx_len_block_size = 64
    hidden_dim_block_size = 128

    ctx_len, hidden_dim = k_quant_int.shape
    ctx_len_block_num = ctx_len // ctx_len_block_size
    hidden_dim_block_num = hidden_dim // hidden_dim_block_size
    k_bit_len = math.ceil(math.log2(k_quant_int.unique().numel()))
    assert k_bit_len == 4

    block = k_quant_int.to(torch.uint8).cpu()
    block_info_buffer = torch.zeros(
        ctx_len_block_num * hidden_dim_block_num,
        2,
        dtype=torch.uint32
    ).cpu()

    compressed_buffer = torch.zeros_like(block)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    compressed_size = k_encode_cpu(
        block,
        compressed_buffer,
        block_info_buffer,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        k_bit_len
    )
    end.record()
    torch.cuda.synchronize()
    compress_time += start.elapsed_time(end)

    return compress_size, compress_time


def v_cpu_compress_gpu_mat_vec_mul(
        v: torch.Tensor,
        config
) -> Tuple[int, float, float, float]:
    # return 0,0,0,0
    v_quant_int, v_quant_zero, v_quant_scale = quant_ints(v, config.block_size, config.v_quant_scale_rel,
                                                          config.quant_method.value[1])
    dtype_ = v.dtype
    v = ((v_quant_int + v_quant_zero) * v_quant_scale).flatten(2, 3)
    v_quant_int = v_quant_int.flatten(2, 3).permute(2, 0, 1, 3).flatten(1, 3).contiguous()
    v_quant_zero = v_quant_zero.flatten()
    v_quant_scale = v_quant_scale.flatten()

    ctx_len_block_size = 128
    hidden_dim_block_size = 64
    head_dim = 128

    ctx_len, hidden_dim = v_quant_int.shape
    ctx_len_block_num = ctx_len // ctx_len_block_size
    hidden_dim_block_num = hidden_dim // hidden_dim_block_size
    v_bit_len = math.ceil(math.log2(v_quant_int.unique().numel()))

    block = v_quant_int.to(torch.uint8).cpu()
    block_info_buffer = torch.zeros(
        ctx_len_block_num * hidden_dim_block_num,
        2,
        dtype=torch.uint32
    ).cpu()

    compressed_buffer = torch.zeros_like(block)

    compressed_size = v_encode_cpu(
        block,
        compressed_buffer,
        block_info_buffer,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        v_bit_len
    )

    decompress_tensor = torch.zeros_like(
        block
    )

    v_decode_cpu(
        compressed_buffer,
        block_info_buffer,
        decompress_tensor,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        v_bit_len
    )

    assert (block == decompress_tensor).all()

    head_num = math.ceil(hidden_dim / head_dim)
    w = torch.randn(
        head_num,
        ctx_len,
        dtype=dtype_
    ).to(v.device).pow(2).softmax(dim=1)

    term1 = torch.zeros(
        # ctx_len_block_num,
        # 1,
        head_num,
        head_dim,
        dtype=torch.float32
    ).to(v.device)

    w_prime = w * v_quant_scale.unsqueeze(0)

    v_our_kernel_time = wv_mat_vec_mul(
        compressed_buffer.to(v.device),
        block_info_buffer.to(v.device),
        w_prime,
        term1,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        v_bit_len
    )

    term1 = term1.to(dtype_)

    # print(f"term1.shape: {term1.shape}, dtype: {term1.dtype}, is_contiguous: {term1.is_contiguous()}")
    # print(f"v_quant_zero.shape: {v_quant_zero.shape}, dtype: {v_quant_zero.dtype}, is_contiguous: {v_quant_zero.is_contiguous()}")
    # print(f"v_quant_scale.shape: {v_quant_scale.shape}, dtype: {v_quant_scale.dtype}, is_contiguous: {v_quant_scale.is_contiguous()}")
    # print(f"w.shape: {w.shape}, dtype: {w.dtype}, is_contiguous: {w.is_contiguous()}")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # term1 = term1.unsqueeze(0)

    our_wv = torch.empty_like(term1)
    v_our_none_kernel_time = fused_wv(
        our_wv, w, v_quant_zero, v_quant_scale, term1
    )

    w_ = w.unsqueeze(1)
    v_ = v.squeeze()

    torch.cuda.synchronize()
    start.record()
    target_wv = (w_ @ v_)
    end.record()
    torch.cuda.synchronize()
    v_pytorch_kernel_time = start.elapsed_time(end)

    target_wv = target_wv.squeeze()

    max_error_percentage = ((target_wv - our_wv).abs() / target_wv.abs().max()).max() * 100

    # print(f"max_error_percentage: {max_error_percentage}")
    if max_error_percentage > 6:
        print(f"max_error_percentage: {max_error_percentage}")
        exit()

    # print(f"v_our_kernel_time: {v_our_kernel_time}, v_our_none_kernel_time: {v_our_none_kernel_time}, v_pytorch_kernel_time: {v_pytorch_kernel_time}")

    # exit()

    return compressed_size, v_our_kernel_time, v_our_none_kernel_time, v_pytorch_kernel_time


def v_cpu_compress(
        v: torch.Tensor,
        config
) -> Tuple[int, float, float, float]:
    compress_size = v.numel() * v.element_size()
    compress_time = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    v_quant_int, v_quant_zero, v_quant_scale = quant_ints(v, config.block_size, config.v_quant_scale_rel,
                                                          config.quant_method.value[1])
    end.record()
    torch.cuda.synchronize()
    compress_time += start.elapsed_time(end)
    dtype_ = v.dtype
    v = ((v_quant_int + v_quant_zero) * v_quant_scale).flatten(2, 3)
    v_quant_int = v_quant_int.flatten(2, 3).permute(2, 0, 1, 3).flatten(1, 3).contiguous()
    v_quant_zero = v_quant_zero.flatten()
    v_quant_scale = v_quant_scale.flatten()

    ctx_len_block_size = 128
    hidden_dim_block_size = 64
    head_dim = 128

    ctx_len, hidden_dim = v_quant_int.shape
    ctx_len_block_num = ctx_len // ctx_len_block_size
    hidden_dim_block_num = hidden_dim // hidden_dim_block_size
    v_bit_len = math.ceil(math.log2(v_quant_int.unique().numel()))

    block = v_quant_int.to(torch.uint8).cpu()
    block_info_buffer = torch.zeros(
        ctx_len_block_num * hidden_dim_block_num,
        2,
        dtype=torch.uint32
    ).cpu()

    compressed_buffer = torch.zeros_like(block)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    compressed_size = v_encode_cpu(
        block,
        compressed_buffer,
        block_info_buffer,
        ctx_len,
        hidden_dim,
        ctx_len_block_size,
        hidden_dim_block_size,
        v_bit_len
    )
    end.record()
    torch.cuda.synchronize()
    compress_time += start.elapsed_time(end)

    return compress_size, compress_time


def throughputs_evaluation_with_data(
        config: PackKVCacheConfig,
        key_caches,
        value_caches
):
    # assert key_caches[0].shape[1] == 32 and key_caches[0].shape[3] == 128
    res = {
        "k_original_size": [],
        "v_original_size": [],
        "k_compressed_size": [],
        "v_compressed_size": [],
        "k_our_kernel_time": [],
        "k_our_none_kernel_time": [],
        "k_pytorch_kernel_time": [],
        "v_our_kernel_time": [],
        "v_our_none_kernel_time": [],
        "v_pytorch_kernel_time": [],
        "k_compress_size": 0,
        "v_compress_size": 0,
        "k_compress_time": 0.0,
        "v_compress_time": 0.0
    }

    layer_num = len(key_caches)

    # k warm up
    for layer_idx in range(3):
        k = key_caches[layer_idx]
        k_cpu_compress_gpu_mat_vec_mul(k, config)

    for layer_idx in range(layer_num):
        k = key_caches[layer_idx]
        # assert k.dtype == torch.float16, "kernel only support fp16 for now"
        k_origin_size = k.numel() * k.element_size()
        res["k_original_size"].append(k_origin_size)
        # k_transform = k.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        # v_transform = v.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        k_compress_size, k_our_kernel_time, k_our_none_kernel_time, k_pytorch_kernel_time = k_cpu_compress_gpu_mat_vec_mul(
            k, config)
        res["k_compressed_size"].append(k_compress_size)
        res["k_our_kernel_time"].append(k_our_kernel_time)
        res["k_our_none_kernel_time"].append(k_our_none_kernel_time)
        res["k_pytorch_kernel_time"].append(k_pytorch_kernel_time)
        # res["k_compress_size"].append(k_compress_size)

    # v warm up
    for layer_idx in range(3):
        v = value_caches[layer_idx]

        v_cpu_compress_gpu_mat_vec_mul(v, config)

    for layer_idx in range(layer_num):
        v = value_caches[layer_idx]
        # assert v.dtype == torch.float16, "kernel only support fp16 for now"
        v_origin_size = v.numel() * v.element_size()
        res["v_original_size"].append(v_origin_size)
        # k_transform = k.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        # v_transform = v.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        v_compress_size, v_our_kernel_time, v_our_none_kernel_time, v_pytorch_kernel_time = v_cpu_compress_gpu_mat_vec_mul(
            v, config)
        res["v_compressed_size"].append(v_compress_size)
        res["v_our_kernel_time"].append(v_our_kernel_time)
        res["v_our_none_kernel_time"].append(v_our_none_kernel_time)
        res["v_pytorch_kernel_time"].append(v_pytorch_kernel_time)
        # res["k_compress_size"].append(k_compress_size)

    return res


def throughputs_evaluation_with_data_rebuttal(
        config: PackKVCacheConfig,
        key_caches,
        value_caches,
        ctx_len_factor
):
    # assert key_caches[0].shape[1] == 32 and key_caches[0].shape[3] == 128
    res = {
        "k_original_size": [],
        "v_original_size": [],
        "k_compressed_size": [],
        "v_compressed_size": [],
        "k_our_kernel_time": [],
        "k_our_none_kernel_time": [],
        "k_pytorch_kernel_time": [],
        "v_our_kernel_time": [],
        "v_our_none_kernel_time": [],
        "v_pytorch_kernel_time": [],
        "k_compress_size": 0,
        "v_compress_size": 0,
        "k_compress_time": 0.0,
        "v_compress_time": 0.0
    }

    for index in range(len(key_caches)):
        key_caches[index] = torch.concat([key_caches[index]]*ctx_len_factor, dim=2)
        value_caches[index] = torch.concat([value_caches[index]]*ctx_len_factor, dim=2)

    layer_num = len(key_caches)

    # k warm up
    for layer_idx in range(3):
        k = key_caches[layer_idx]
        k_cpu_compress_gpu_mat_vec_mul(k, config)

    for layer_idx in range(layer_num):
        k = key_caches[layer_idx]
        # assert k.dtype == torch.float16, "kernel only support fp16 for now"
        k_origin_size = k.numel() * k.element_size()
        res["k_original_size"].append(k_origin_size)
        # k_transform = k.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        # v_transform = v.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        k_compress_size, k_our_kernel_time, k_our_none_kernel_time, k_pytorch_kernel_time = k_cpu_compress_gpu_mat_vec_mul(
            k, config)
        res["k_compressed_size"].append(k_compress_size)
        res["k_our_kernel_time"].append(k_our_kernel_time)
        res["k_our_none_kernel_time"].append(k_our_none_kernel_time)
        res["k_pytorch_kernel_time"].append(k_pytorch_kernel_time)
        # res["k_compress_size"].append(k_compress_size)

    # v warm up
    for layer_idx in range(3):
        v = value_caches[layer_idx]

        v_cpu_compress_gpu_mat_vec_mul(v, config)

    for layer_idx in range(layer_num):
        v = value_caches[layer_idx]
        # assert v.dtype == torch.float16, "kernel only support fp16 for now"
        v_origin_size = v.numel() * v.element_size()
        res["v_original_size"].append(v_origin_size)
        # k_transform = k.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        # v_transform = v.permute(2, 0, 1, 3).flatten(1,3).contiguous()
        v_compress_size, v_our_kernel_time, v_our_none_kernel_time, v_pytorch_kernel_time = v_cpu_compress_gpu_mat_vec_mul(
            v, config)
        res["v_compressed_size"].append(v_compress_size)
        res["v_our_kernel_time"].append(v_our_kernel_time)
        res["v_our_none_kernel_time"].append(v_our_none_kernel_time)
        res["v_pytorch_kernel_time"].append(v_pytorch_kernel_time)
        # res["k_compress_size"].append(k_compress_size)

    return res

def compressor_throughputs_evaluation_with_data(
        config: PackKVCacheConfig,
        key_caches,
        value_caches
):
    # assert key_caches[0].shape[1] == 32 and key_caches[0].shape[3] == 128
    res = {
        "k_compress_size": [],
        "v_compress_size": [],
        "k_compress_time": [],
        "v_compress_time": []
    }

    layer_num = len(key_caches)

    # k warm up
    for layer_idx in range(3):
        k = key_caches[layer_idx]
        k_cpu_compress(k, config)

    for layer_idx in range(layer_num):
        k = key_caches[layer_idx]
        k_compress_size, k_compress_time = k_cpu_compress(
            k, config)
        res["k_compress_size"].append(k_compress_size)
        res["k_compress_time"].append(k_compress_time)

    # v warm up
    for layer_idx in range(3):
        v = value_caches[layer_idx]

        v_cpu_compress(v, config)

    for layer_idx in range(layer_num):
        v = value_caches[layer_idx]
        v_compress_size, v_compress_time = v_cpu_compress(v, config)
        res["v_compress_size"].append(v_compress_size)
        res["v_compress_time"].append(v_compress_time)

    return res


def throughput_evaluation(
        config: PackKVCacheConfig,
        ctx_len: int,
        enable_save: bool,
        logger,
        collect_round: int = 1,
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    logger.info(f"Created a temp config for cache collection and set its enable_quant to False.")
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,
        model_name=config.model_name,
        quant_method=config.quant_method,
        repack_method=config.repack_method,
        high_precision_zero_point=config.high_precision_zero_point,
        block_size=config.block_size,
        buffer_size=config.buffer_size,
        pack_size=config.pack_size,
        k_quant_scale_rel=config.k_quant_scale_rel,
        v_quant_scale_rel=config.v_quant_scale_rel
    )

    PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        PackKVCacheConfigStatic.config.model_name,
        torch_dtype="auto",
        device_map="auto",
        # attn_implementation="sdpa"
    )
    # batch_size = 1
    # lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    inputs = get_ctx_len_text_from_wikitext_103_v1(ctx_len, tokenizer).to(model.device)

    with torch.no_grad():
        _ = model(**inputs)
    PackKVCachePytorchQuant.round_ = 0

    if enable_save:
        save_extract_cache(config.model_name, ctx_len, PackKVCacheConfigStatic.extract_cache, "./dumped_cache")

    rts = []

    logger.info(f"Cache Size {PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024 / 1024} GB")

    for round_i in range(collect_round):
        key_caches = list(PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values())
        value_caches = list(PackKVCacheConfigStatic.extract_cache.value_caches[round_i].values())
        rts.append(throughputs_evaluation_with_data(config, key_caches, value_caches))

    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None
    return rts


def eager_attention_forward(
        # query: torch.Tensor,
        key: List[torch.Tensor],
        value: List[torch.Tensor],
        # scaling: float,
):
    other_times = []
    b, h, c, hd = key[0].shape
    query = torch.randn(b, h, 1, hd, device=key[0].device, dtype=key[0].dtype)
    for key, value in zip(key, value):
        time = 0
        key = key.transpose(2, 3)

        attn_weights = torch.matmul(query, key)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        attn_weights = attn_weights * 11.31
        end.record()
        torch.cuda.synchronize()
        time += start.elapsed_time(end)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        end.record()
        torch.cuda.synchronize()
        time += start.elapsed_time(end)

        attn_weights = attn_weights.to(value.dtype)

        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous()
        other_times.append(time)
    return other_times


def throughput_evaluation_rebuttal(
        config: PackKVCacheConfig,
        ctx_len: int,
        enable_save: bool,
        logger,
        collect_round: int = 1,
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    logger.info(f"Created a temp config for cache collection and set its enable_quant to False.")
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,
        model_name=config.model_name,
        quant_method=config.quant_method,
        repack_method=config.repack_method,
        high_precision_zero_point=config.high_precision_zero_point,
        block_size=config.block_size,
        buffer_size=config.buffer_size,
        pack_size=config.pack_size,
        k_quant_scale_rel=config.k_quant_scale_rel,
        v_quant_scale_rel=config.v_quant_scale_rel
    )

    PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        PackKVCacheConfigStatic.config.model_name,
        torch_dtype="auto",
        device_map="auto",
        # attn_implementation="sdpa"
    )
    # batch_size = 1
    # lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    real_ctx_len = 4 * 1024 if ctx_len >= 4 * 1024 else ctx_len
    assert ctx_len % real_ctx_len == 0
    ctx_len_factor = ctx_len // real_ctx_len
    inputs = get_ctx_len_text_from_wikitext_103_v1(real_ctx_len, tokenizer).to(model.device)

    with torch.no_grad():
        _ = model(**inputs)
    PackKVCachePytorchQuant.round_ = 0

    if enable_save:
        save_extract_cache(config.model_name, ctx_len, PackKVCacheConfigStatic.extract_cache, "./dumped_cache")

    rts = []

    logger.info(f"Cache Size {PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024 / 1024 * ctx_len_factor} GB")

    for round_i in range(collect_round):
        key_caches = list(PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values())
        value_caches = list(PackKVCacheConfigStatic.extract_cache.value_caches[round_i].values())
        other_overhead = eager_attention_forward(key_caches, value_caches)
        rts.append([throughputs_evaluation_with_data_rebuttal(config, key_caches, value_caches, ctx_len_factor), other_overhead])

    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None
    return rts


def end_to_end_simulation_rebuttal(
        config: PackKVCacheConfig,
        eager_attention_speed_up: float,
        ctx_len: int,
        logger,
        batch_size: int = 1,
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    model_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    logger.info(f"Creating {batch_size} prompts with target length {ctx_len} tokens...")
    base_prompt = "Hello, who are you? I am an AI assistant. "

    # First, tokenize base_prompt to get its actual token count
    base_tokens = tokenizer(base_prompt, return_tensors="pt").input_ids
    actual_tokens_per_repeat = base_tokens.shape[1]
    logger.info(f"Base prompt has {actual_tokens_per_repeat} tokens")

    # Calculate how many repetitions we need to reach ctx_len
    repeat_count = max(1, ctx_len // actual_tokens_per_repeat)
    logger.info(f"Repeating {repeat_count} times to reach ~{repeat_count * actual_tokens_per_repeat} tokens")

    prompts = [base_prompt * repeat_count for _ in range(batch_size)]
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=ctx_len,
    ).to(model.device)

    actual_seq_len = inputs.input_ids.shape[1]
    logger.info(
        f"Created batch with shape: {inputs.input_ids.shape} (batch_size={batch_size}, seq_len={actual_seq_len})")

    past_key_values = None
    current_input_ids = None
    cache_position = None

    def initialize_kv_cache():
        global past_key_values, current_input_ids, cache_position

        print(f"Creating StaticCache (batch_size={batch_size}, max_length={ctx_len + 100})...")
        # Create static cache with room for prefill + decode tokens
        past_key_values = StaticCache(
            config=model.config,
            max_batch_size=batch_size,
            max_cache_len=ctx_len + 100,  # Extra room for decode tokens
            device=model.device,
            dtype=torch.float16
        )

        print(f"Initializing KV cache with prefill (batch_size={batch_size})...")
        seq_len = inputs.input_ids.shape[1]
        cache_position = torch.arange(0, seq_len, device=model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                logits_to_keep=1  # solve prefill phase OOM
            )
            # past_key_values is updated in-place
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            current_input_ids = next_token_id
            # Update cache position for next decode step
            cache_position = torch.tensor([seq_len], device=model.device)

        logger.info(f"KV cache initialized. Current position: {cache_position.item()}")

    # Show GPU memory after prefill
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"GPU Memory after prefill: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    new_token = 10

    initialize_kv_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager"
    )

    def decode_func():
        global past_key_values, current_input_ids, cache_position
        generated_tokens = []

        with torch.no_grad():
            for i in range(new_token):
                outputs = model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True
                )
                # past_key_values is updated in-place, no need to reassign
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                current_input_ids = next_token_id
                generated_tokens.append(next_token_id)
                # Increment cache position
                cache_position = cache_position + 1

        all_tokens = torch.cat(generated_tokens, dim=1)
        # Show first sample in batch
        decoded_text = tokenizer.decode(all_tokens[0], skip_special_tokens=True)
        print(f"Generated {new_token} tokens for batch_size={batch_size}")
        print(f"Sample from first item in batch: {decoded_text}")

    kernel_info, avg_func_time = profile_func(decode_func, warmup_runs=3, profile_runs=5)

    return kernel_info, avg_func_time
    # print("\nReturned kernel info:")
    # print(f"Number of kernels: {len(kernel_info['kernel_names'])}")
    # print(f"Total GPU time: {kernel_info['total_time']:.2f} us")
    # print(f"Time per token: {kernel_info['total_time'] / (batch_size * new_token):.2f} us")

def compressor_throughput_evaluation_rebuttal(
        config: PackKVCacheConfig,
        ctx_len: int,
        enable_save: bool,
        logger,
        collect_round: int = 1,
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    logger.info(f"Created a temp config for cache collection and set its enable_quant to False.")
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,
        model_name=config.model_name,
        quant_method=config.quant_method,
        repack_method=config.repack_method,
        high_precision_zero_point=config.high_precision_zero_point,
        block_size=config.block_size,
        buffer_size=config.buffer_size,
        pack_size=config.pack_size,
        k_quant_scale_rel=config.k_quant_scale_rel,
        v_quant_scale_rel=config.v_quant_scale_rel
    )

    PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        PackKVCacheConfigStatic.config.model_name,
        torch_dtype="auto",
        device_map="auto",
        # attn_implementation="sdpa"
    )
    # batch_size = 1
    # lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    inputs = get_ctx_len_text_from_wikitext_103_v1(ctx_len, tokenizer).to(model.device)

    with torch.no_grad():
        _ = model(**inputs)
    PackKVCachePytorchQuant.round_ = 0

    if enable_save:
        save_extract_cache(config.model_name, ctx_len, PackKVCacheConfigStatic.extract_cache, "./dumped_cache")

    rts = []

    logger.info(f"Cache Size {PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024 / 1024} GB")
    size_mb = PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024

    for round_i in range(collect_round):
        key_caches = list(PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values())
        value_caches = list(PackKVCacheConfigStatic.extract_cache.value_caches[round_i].values())
        rts.append(compressor_throughputs_evaluation_with_data(config, key_caches, value_caches))

    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None
    return rts, size_mb


def cr_evaluation_detail_rebuttal(
        config: PackKVCacheConfig,
        ctx_len: int,
        enable_save: bool,
        logger,
        collect_round: int = 1,
        before_and_after_repacking=None
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    # logger.info(f"Created a temp config for cache collection and set its enable_quant to False.")
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,
        model_name=config.model_name,
        quant_method=config.quant_method,
        repack_method=config.repack_method,
        high_precision_zero_point=config.high_precision_zero_point,
        block_size=config.block_size,
        buffer_size=config.buffer_size,
        pack_size=config.pack_size,
        k_quant_scale_rel=config.k_quant_scale_rel,
        v_quant_scale_rel=config.v_quant_scale_rel
    )

    PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        PackKVCacheConfigStatic.config.model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    # batch_size = 1
    # lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    inputs = get_ctx_len_text_from_wikitext_103_v1(ctx_len, tokenizer).to(model.device)

    with torch.no_grad():
        _ = model(**inputs)
    PackKVCachePytorchQuant.round_ = 0

    if enable_save:
        save_extract_cache(config.model_name, ctx_len, PackKVCacheConfigStatic.extract_cache, "./dumped_cache")

    rts = []

    logger.info(f"Cache Size {PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024 / 1024} GB")

    for round_i in range(collect_round):
        key_caches = list(PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values())
        value_caches = list(PackKVCacheConfigStatic.extract_cache.value_caches[round_i].values())
        rts.append(
            crs_evaluation_with_data_detail_rebuttal(config, key_caches, value_caches, before_and_after_repacking))

    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None
    return rts


def repacking_throughput_rebuttal(
        config: PackKVCacheConfig,
        ctx_len: int,
        enable_save: bool,
        logger,
        collect_round: int = 1,
        before_and_after_repacking=None
):
    logger.info(f"ctx_len: {ctx_len}")
    logger.info(config)
    # logger.info(f"Created a temp config for cache collection and set its enable_quant to False.")
    PackKVCacheConfigStatic.config = PackKVCacheConfig(
        enable_quant=False,
        model_name=config.model_name,
        quant_method=config.quant_method,
        repack_method=config.repack_method,
        high_precision_zero_point=config.high_precision_zero_point,
        block_size=config.block_size,
        buffer_size=config.buffer_size,
        pack_size=config.pack_size,
        k_quant_scale_rel=config.k_quant_scale_rel,
        v_quant_scale_rel=config.v_quant_scale_rel
    )

    PackKVCacheConfigStatic.extract_cache = ExtractCacheConfig(collect_round)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_class = MODEL_CLASS_MAP.get(config.model_name)
    if model_class is None:
        raise ValueError(f"Model class not found for {config.model_name}")
    model = model_class.from_pretrained(
        PackKVCacheConfigStatic.config.model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    # batch_size = 1
    # lm_eval_warp = LMEvalWrapper(model, tokenizer, batch_size)
    inputs = get_ctx_len_text_from_wikitext_103_v1(ctx_len, tokenizer).to(model.device)

    with torch.no_grad():
        _ = model(**inputs)
    PackKVCachePytorchQuant.round_ = 0

    if enable_save:
        save_extract_cache(config.model_name, ctx_len, PackKVCacheConfigStatic.extract_cache, "./dumped_cache")

    rts = []

    logger.info(f"Cache Size {PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024 / 1024} GB")
    size_mb = PackKVCacheConfigStatic.extract_cache.size() / 8 / 1024 / 1024

    for round_i in range(collect_round):
        key_caches = list(PackKVCacheConfigStatic.extract_cache.key_caches[round_i].values())
        value_caches = list(PackKVCacheConfigStatic.extract_cache.value_caches[round_i].values())
        rts.append(repacking_throughput_with_kv_rebuttal(config, key_caches, value_caches))

    PackKVCacheConfigStatic.config = None
    PackKVCacheConfigStatic.extract_cache = None
    return rts, size_mb
