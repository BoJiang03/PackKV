#! /usr/bin/env python
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.compute import QuantMode, QuantMethod, RepackMethod
from utils.config import PackKVCacheConfig
from utils.serialization import save, unified_hash
from utils.util import get_logger, block_other_logger
import numpy as np
import pickle

logger = get_logger(__file__)
block_other_logger(logger)

BLOCK_SIZE = 64
BUFFER_SIZE = 128 + 64

model_list = [
    "meta-llama/Llama-2-13b-hf",
    "Qwen/Qwen3-8B"
]

benchmark = "gsm8k"

channel_quant_k = 0.25
token_quant_k = 0.15
token_quant_v = 0.20

def gen_quant_rels(turning_point):
    quant_rels = [0.01, turning_point * 0.4, turning_point * 0.6, turning_point * 0.8, turning_point, turning_point * 1.2]
    return quant_rels

channel_quant_k_quant_scale_rels = [0.001] +  gen_quant_rels(channel_quant_k) + [channel_quant_k * 1.4, channel_quant_k * 1.6]
token_quant_k_quant_scale_rels = gen_quant_rels(token_quant_k)
token_quant_v_quant_scale_rels = gen_quant_rels(token_quant_v)

# scale_num = 5
setting_map = {}

print(channel_quant_k_quant_scale_rels)
print(token_quant_k_quant_scale_rels)
print(token_quant_v_quant_scale_rels)

for model_name in model_list:
    for quant_scale in channel_quant_k_quant_scale_rels:
        # k channel quant
        config = PackKVCacheConfig(
                    enable_quant=True,
                    model_name=model_name,
                    quant_method=QuantMethod.KIVI,
                    repack_method=RepackMethod.NONE,
                    high_precision_zero_point=True,
                    block_size=BLOCK_SIZE,
                    buffer_size=BUFFER_SIZE,
                    pack_size=16,
                    k_quant_scale_rel=quant_scale,
                    v_quant_scale_rel=0.01
        )
        pair = (benchmark, config)
        if unified_hash(pair) in setting_map:
            assert pair == setting_map[unified_hash(pair)], "config with same hash but different value"
        else:
            setting_map[unified_hash(pair)] = pair
        config = PackKVCacheConfig(
                    enable_quant=True,
                    model_name=model_name,
                    quant_method=QuantMethod.KIVI,
                    repack_method=RepackMethod.NONE,
                    high_precision_zero_point=False,
                    block_size=BLOCK_SIZE,
                    buffer_size=BUFFER_SIZE,
                    pack_size=16,
                    k_quant_scale_rel=quant_scale,
                    v_quant_scale_rel=0.01
        )
        pair = (benchmark, config)
        if unified_hash(pair) in setting_map:
            assert pair == setting_map[unified_hash(pair)], "config with same hash but different value"
        else:
            setting_map[unified_hash(pair)] = pair
    for quant_scale in token_quant_k_quant_scale_rels:
        # k token quant
        config = PackKVCacheConfig(
                    enable_quant=True,
                    model_name=model_name,
                    quant_method=QuantMethod.PackKV,
                    repack_method=RepackMethod.NONE,
                    high_precision_zero_point=True,
                    block_size=BLOCK_SIZE,
                    buffer_size=BUFFER_SIZE,
                    pack_size=16,
                    k_quant_scale_rel=quant_scale,
                    v_quant_scale_rel=0.01
        )
        pair = (benchmark, config)
        if unified_hash(pair) in setting_map:
            assert pair == setting_map[unified_hash(pair)], "config with same hash but different value"
        else:
            setting_map[unified_hash(pair)] = pair
        config = PackKVCacheConfig(
                    enable_quant=True,
                    model_name=model_name,
                    quant_method=QuantMethod.PackKV,
                    repack_method=RepackMethod.NONE,
                    high_precision_zero_point=False,
                    block_size=BLOCK_SIZE,
                    buffer_size=BUFFER_SIZE,
                    pack_size=16,
                    k_quant_scale_rel=quant_scale,
                    v_quant_scale_rel=0.01
        )
        pair = (benchmark, config)
        if unified_hash(pair) in setting_map:
            assert pair == setting_map[unified_hash(pair)], "config with same hash but different value"
        else:
            setting_map[unified_hash(pair)] = pair
    for quant_scale in token_quant_v_quant_scale_rels:
        # v token quant
        config = PackKVCacheConfig(
                    enable_quant=True,
                    model_name=model_name,
                    quant_method=QuantMethod.PackKV,
                    repack_method=RepackMethod.NONE,
                    high_precision_zero_point=True,
                    block_size=BLOCK_SIZE,
                    buffer_size=BUFFER_SIZE,
                    pack_size=16,
                    k_quant_scale_rel=0.01,
                    v_quant_scale_rel=quant_scale
        )
        pair = (benchmark, config)
        if unified_hash(pair) in setting_map:
            assert pair == setting_map[unified_hash(pair)], "config with same hash but different value"
        else:
            setting_map[unified_hash(pair)] = pair
        config = PackKVCacheConfig(
                    enable_quant=True,
                    model_name=model_name,
                    quant_method=QuantMethod.PackKV,
                    repack_method=RepackMethod.NONE,
                    high_precision_zero_point=False,
                    block_size=BLOCK_SIZE,
                    buffer_size=BUFFER_SIZE,
                    pack_size=16,
                    k_quant_scale_rel=0.01,
                    v_quant_scale_rel=quant_scale
        )
        pair = (benchmark, config)
        if unified_hash(pair) in setting_map:
            assert pair == setting_map[unified_hash(pair)], "config with same hash but different value"
        else:
            setting_map[unified_hash(pair)] = pair

setting_path = "data/zero_point/zero_point_setting_map.pkl"
print(setting_map)
save(setting_map, setting_path)
logger.info(f"Setting map saved to {setting_path}, total {len(setting_map)} entries")