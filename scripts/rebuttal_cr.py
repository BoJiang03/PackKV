#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os

import numpy as np

from utils.compute import QuantMethod, RepackMethod
from utils.util import get_logger, block_other_logger

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import cr_evaluation
from utils.config import PackKVCacheConfig
from utils.serialization import load, save

logger = get_logger(__file__)
block_other_logger(logger)

ctx_len =  1024 * 4
model_name = "meta-llama/Llama-2-13b-hf"

BLOCK_SIZE = 64
BUFFER_SIZE = 128 + 64

k_cr_improves = []
v_cr_improves = []

for scale in range(1, 20, 2):
    scale = scale / 100
    config = PackKVCacheConfig(
        model_name=model_name,
        quant_method=QuantMethod.PackKV,
        repack_method=RepackMethod.GREEDY,
        high_precision_zero_point=False,
        block_size=BLOCK_SIZE,
        buffer_size=BUFFER_SIZE,
        pack_size=8,
        k_quant_scale_rel= scale,
        v_quant_scale_rel= scale,
    )
    cr_result = cr_evaluation(
        config=config,
        ctx_len=ctx_len,
        enable_save=False,
        logger=logger
    )
    cr_result = cr_result[0]

    k_quant_size = sum(cr_result["k_quant_size"])
    v_quant_size = sum(cr_result["v_quant_size"])
    k_our_size = sum(cr_result["k_encode_size_after_repack"])
    v_our_size = sum(cr_result["v_encode_size_after_repack"])

    k_cr_improves.append(k_quant_size / k_our_size)
    v_cr_improves.append(v_quant_size / v_our_size)


k_cr_improve = np.mean(k_cr_improves).round(2)
v_cr_improve = np.mean(v_cr_improves).round(2)

print(f"k_cr_improves: {k_cr_improves}")
print(f"v_cr_improves: {v_cr_improves}")
print(f"k_cr_improve: {k_cr_improve}")
print(f"v_cr_improve: {v_cr_improve}")