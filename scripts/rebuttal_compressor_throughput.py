#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os

import numpy as np

from utils.compute import QuantMethod, RepackMethod
from utils.util import get_logger, block_other_logger

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import compressor_throughput_evaluation_rebuttal
from utils.config import PackKVCacheConfig
from utils.serialization import load, save

logger = get_logger(__file__)
block_other_logger(logger)

ctx_len =  1024 * 4
model_name = "meta-llama/Llama-2-13b-hf"
BLOCK_SIZE = 64
BUFFER_SIZE = 128 + 64

config = PackKVCacheConfig(
    model_name=model_name,
    quant_method=QuantMethod.PackKV,
    repack_method=RepackMethod.GREEDY,
    high_precision_zero_point=False,
    block_size=BLOCK_SIZE,
    buffer_size=BUFFER_SIZE,
    pack_size=8,
    k_quant_scale_rel=0.14,
    v_quant_scale_rel=0.2,
)
cr_result, size_mb = compressor_throughput_evaluation_rebuttal(
    config=config,
    ctx_len=ctx_len,
    enable_save=False,
    logger=logger
)
cr_result = cr_result[0]

k_size_sum = sum(cr_result["k_compress_size"]) / 1024 / 1024
v_size_sum = sum(cr_result["v_compress_size"]) / 1024 / 1024
k_time_sum = sum(cr_result["k_compress_time"]) / 1000
v_time_sum = sum(cr_result["v_compress_time"]) / 1000

k_compress_throughput = k_size_sum / k_time_sum
v_compress_throughput = v_size_sum / v_time_sum

print(f"k_compress_throughput: {k_compress_throughput:.2f} MB/s")
print(f"v_compress_throughput: {v_compress_throughput:.2f} MB/s")