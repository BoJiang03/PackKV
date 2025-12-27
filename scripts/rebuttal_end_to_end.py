#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os

import numpy as np

from utils.compute import QuantMethod, RepackMethod

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import throughput_evaluation, throughput_evaluation_rebuttal, end_to_end_simulation_rebuttal
from utils.config import PackKVCacheConfig
from utils.util import get_logger, block_other_logger, register_notify
import torch

# Our current implementation only supports fused decompression with matrix–vector multiplication. Therefore, the fairest comparison is against an eager attention implementation.

# In addition, we can only use MHA models because, in the transformers package, GQA triggers multiple memory copy operations every time eager attention is called(because repack_kv func call). As a result, memory copy overhead dominates GPU time, which is essentially a limitation of the transformers library (it is not a real serving framework).

# However, the MHA models available in my model list support a maximum context length of only 4K tokens. Consequently, to simulate longer context lengths, I have to duplicate the KV cache multiple times. If you really want to test long context, you can try codellama or yarn llama2

logger = get_logger(__file__)
block_other_logger(logger)

model_name = "meta-llama/Llama-2-13b-hf"

ctx_len_list = [
    128,
    # 1024,
    # 2048,
    # 4096,
    8192,
    16384,
    32768,
    # 65536,
    # 131072
]

BUFFER_SIZE = 128 + 64
BLOCK_SIZE = 64
PACK_SIZE = 16

end_to_end_simulation_speedups = []

kernel_infos = []

for ctx_len in ctx_len_list:
    config = PackKVCacheConfig(
        enable_quant=False,
        model_name=model_name,
        quant_method=QuantMethod.PackKV,
        repack_method=RepackMethod.NONE,
        high_precision_zero_point=False,
        block_size=BLOCK_SIZE,
        buffer_size=BUFFER_SIZE,
        pack_size=PACK_SIZE,
        k_quant_scale_rel=0.1,
        v_quant_scale_rel=0.2,
    )

    pair = (ctx_len, config)

    result = throughput_evaluation_rebuttal(config, ctx_len, enable_save=False, logger=logger)
    result = result[0]
    core_kernel_time = result[0]
    k_our_kernel_time = sum(core_kernel_time["k_our_kernel_time"])
    v_our_kernel_time = sum(core_kernel_time["v_our_kernel_time"])
    k_pytorch_kernel_time = sum(core_kernel_time["k_pytorch_kernel_time"])
    v_pytorch_kernel_time = sum(core_kernel_time["v_pytorch_kernel_time"])
    other_overhead = sum(result[1])
    torch.cuda.empty_cache()
    eager_attention_speed_up_ = (k_pytorch_kernel_time + v_pytorch_kernel_time + other_overhead) / (
                k_our_kernel_time + v_our_kernel_time + other_overhead)
    kernel_info, avg_func_time = end_to_end_simulation_rebuttal(config,eager_attention_speed_up_, ctx_len, logger=logger)
    torch.cuda.empty_cache()
    kernel_infos.append((kernel_info, avg_func_time, eager_attention_speed_up_))

base_time = kernel_infos[0][1] # ms, the time we spend on weight and other thing

end_to_end_speedups = []

for index, ctx_len in enumerate(ctx_len_list):
    if index == 0: continue
    avg_func_time = kernel_infos[index][1]

    delta_time = avg_func_time - base_time # the time we spend on extra kv cache, because when ctx len increase the only computation can increase is the kv cache matrix vector multiplication

    speedup_delta_time = delta_time / kernel_infos[index][2]

    new_sum_time = base_time + speedup_delta_time

    end_to_end_speedups.append(avg_func_time / new_sum_time)

print(np.mean(end_to_end_speedups))