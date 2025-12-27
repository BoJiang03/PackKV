#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os

import numpy as np

from utils.compute import QuantMethod, RepackMethod
from utils.util import get_logger, block_other_logger

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import repacking_throughput_rebuttal
from utils.config import PackKVCacheConfig
from utils.serialization import load, save

logger = get_logger(__file__)
block_other_logger(logger)

ctx_len =  1024 * 4
model_name = "meta-llama/Llama-2-13b-hf"

BLOCK_SIZE = 64
BUFFER_SIZE = 128 + 64
quant_scale = 0.2

config = PackKVCacheConfig(
    model_name=model_name,
    quant_method=QuantMethod.PackKV,
    repack_method=RepackMethod.GREEDY,
    high_precision_zero_point=False,
    block_size=BLOCK_SIZE,
    buffer_size=BUFFER_SIZE,
    pack_size=8,
    k_quant_scale_rel=quant_scale,
    v_quant_scale_rel=quant_scale,
)

generate_throughput_mbs = 40 * 5120 * 2 * 50 * 2 / 1024 / 1024

throughput_result, size_mb = repacking_throughput_rebuttal(
    config=config,
    ctx_len=ctx_len,
    enable_save=False,
    logger=logger
)
throughput_result = throughput_result[0]

greedy_repacking_time = sum(throughput_result["greedy_repacking_time"])
median_repacking_time = sum(throughput_result["median_repacking_time"])

greedy_repacking_throughput = size_mb / 1024 / (greedy_repacking_time / 1000)
median_repacking_throughput = size_mb / 1024 / (median_repacking_time / 1000)
generate_time = size_mb / generate_throughput_mbs
greedy_repacking_time_percent = (greedy_repacking_time / 1000) / generate_time
median_repacking_time_percent = (median_repacking_time / 1000) / generate_time

print(f"greedy_repacking_time_percent: {greedy_repacking_time_percent:.2f}")
print(f"median_repacking_time_percent: {median_repacking_time_percent:.2f}")
print(f"greedy_repacking_throughput: {greedy_repacking_throughput:.2f} GB/s")
print(f"median_repacking_throughput: {median_repacking_throughput:.2f} GB/s")
print(f"generate_time: {generate_time:.2f} s")