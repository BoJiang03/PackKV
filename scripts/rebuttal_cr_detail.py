#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os

import numpy as np

from utils.compute import QuantMethod, RepackMethod
from utils.util import get_logger, block_other_logger

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import cr_evaluation, cr_evaluation_detail_rebuttal
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
cr_result = cr_evaluation_detail_rebuttal(
    config=config,
    ctx_len=ctx_len,
    enable_save=False,
    logger=logger
)
cr_result = cr_result[0]

print("Pack size = 8")

for name in ["k","v"]:
    quant_zero_size = sum(cr_result[name + "_quant_zero_point_size"])
    quant_scale_size = sum(cr_result[name + "_quant_scale_size"])
    bitpack_min_value_size = sum(cr_result[name + "_bitpack_min_value_size"])
    bitpack_encode_len_size = sum(cr_result[name + "_bitpack_encode_len_size"])
    bitpack_encoded_size = sum(cr_result[name + "_bitpack_encoded_size"])
    all_sum = quant_zero_size + quant_scale_size + bitpack_min_value_size + bitpack_encode_len_size + bitpack_encoded_size

    payload = bitpack_encoded_size
    metadata = quant_zero_size + quant_scale_size + bitpack_min_value_size + bitpack_encode_len_size

    print(f"{name} all size: {all_sum}")
    print(f"{name} payload: {payload}, percentage: {payload / all_sum}")
    print(f"{name} metadata: {metadata}, percentage: {metadata / all_sum}")

print("\n\n\nPack size = 16")

config = PackKVCacheConfig(
    model_name=model_name,
    quant_method=QuantMethod.PackKV,
    repack_method=RepackMethod.GREEDY,
    high_precision_zero_point=False,
    block_size=BLOCK_SIZE,
    buffer_size=BUFFER_SIZE,
    pack_size=16,
    k_quant_scale_rel=quant_scale,
    v_quant_scale_rel=quant_scale,
)
cr_result = cr_evaluation_detail_rebuttal(
    config=config,
    ctx_len=ctx_len,
    enable_save=False,
    logger=logger
)
cr_result = cr_result[0]

for name in ["k","v"]:
    quant_zero_size = sum(cr_result[name + "_quant_zero_point_size"])
    quant_scale_size = sum(cr_result[name + "_quant_scale_size"])
    bitpack_min_value_size = sum(cr_result[name + "_bitpack_min_value_size"])
    bitpack_encode_len_size = sum(cr_result[name + "_bitpack_encode_len_size"])
    bitpack_encoded_size = sum(cr_result[name + "_bitpack_encoded_size"])
    all_sum = quant_zero_size + quant_scale_size + bitpack_min_value_size + bitpack_encode_len_size + bitpack_encoded_size

    payload = bitpack_encoded_size
    metadata = quant_zero_size + quant_scale_size + bitpack_min_value_size + bitpack_encode_len_size

    print(f"{name} all size: {all_sum}")
    print(f"{name} payload: {payload}, percentage: {payload / all_sum}")
    print(f"{name} metadata: {metadata}, percentage: {metadata / all_sum}")