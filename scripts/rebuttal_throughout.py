#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os

import numpy as np

from utils.compute import QuantMethod, RepackMethod

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import throughput_evaluation, throughput_evaluation_rebuttal
from utils.config import PackKVCacheConfig
from utils.serialization import load, save
from utils.util import get_logger, block_other_logger, register_notify
from utils.gpu_scheduler import run_multi_gpu_tasks
import torch
from torch import nn

logger = get_logger(__file__)
block_other_logger(logger)

model_list = [
    "meta-llama/Llama-3.1-8B",
    # "mistralai/Ministral-8B-Instruct-2410",
]

ctx_len_list = [
    # 1024,
    # 2048,
    # 4096,
    # 8192,
    16384,
    32768,
    65536,
    131072
]

BUFFER_SIZE = 128 + 64
BLOCK_SIZE = 64
PACK_SIZE = 16

eager_attention_speed_up = []

for model_name in model_list:
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

        eager_attention_speed_up.append((k_pytorch_kernel_time + v_pytorch_kernel_time + other_overhead) / (k_our_kernel_time + v_our_kernel_time + other_overhead))

# print mean of eager_attention_speed_up
print(np.mean(eager_attention_speed_up).round(2))