#! /usr/bin/env python
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.compute import QuantMethod, RepackMethod
from utils.config import PackKVCacheConfig
from utils.serialization import save, unified_hash
from utils.util import get_logger, block_other_logger
from evaluation.evaluation import accuracy_evaluation, cr_evaluation, throughput_evaluation

logger = get_logger(__file__)
block_other_logger(logger)
# register_notify()

model_list = [
    "meta-llama/Llama-3.1-8B",
    "mistralai/Ministral-8B-Instruct-2410",
]

ctx_len_list = [
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072
]

BUFFER_SIZE = 128 + 64
BLOCK_SIZE = 64
PACK_SIZE = 16

def add_to_map(pair, setting_map):
    if unified_hash(pair) in setting_map:
        assert pair == setting_map[unified_hash(pair)], "config with same hash but different value"
    else:
        setting_map[unified_hash(pair)] = pair

setting_map = {}

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
        add_to_map(pair, setting_map)

setting_path = "data/throughput/throughput_setting_map_rtx_pro.pkl"
save(setting_map, setting_path)
logger.info(f"Setting map saved to {setting_path}, total {len(setting_map)} entries")