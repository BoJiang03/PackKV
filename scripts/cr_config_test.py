#! /usr/bin/env python
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.compute import QuantMethod, RepackMethod
from utils.config import PackKVCacheConfig
from evaluation.evaluation import cr_evaluation
from utils.util import get_logger, block_other_logger

BLOCK_SIZE = 64
BUFFER_SIZE = 128 + 64
ctx_len = 8192

config = PackKVCacheConfig(
                        enable_quant=True,
                        model_name="meta-llama/Llama-2-13b-hf",
                        quant_method=QuantMethod.PackKV,
                        repack_method=RepackMethod.GREEDY,
                        high_precision_zero_point=False,
                        block_size=BLOCK_SIZE,
                        buffer_size=BUFFER_SIZE,
                        pack_size=16,
                        k_quant_scale_rel=0.1,
                        v_quant_scale_rel=0.2
            )

logger = get_logger(__file__)
block_other_logger(logger)

before_and_after = []
cr_result = cr_evaluation(
        config=config,
        ctx_len=ctx_len,
        enable_save=False,
        logger=logger
    )

