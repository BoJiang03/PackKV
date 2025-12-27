#! /usr/bin/env python
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.compute import QuantMode, QuantMethod, RepackMethod
from utils.config import PackKVCacheConfig
from utils.serialization import save, unified_hash
from utils.util import get_logger, block_other_logger

logger = get_logger(__file__)
block_other_logger(logger)

BLOCK_SIZE = 64
BUFFER_SIZE = 128 + 64

model_list = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-2-13b-hf",
    # "meta-llama/Llama-2-7b-hf",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "mistralai/Ministral-8B-Instruct-2410",
    "microsoft/phi-4"
]

k_rel_quant_scale = 0.1
v_rel_quant_scale = 0.2

def add_to_map(pair, setting_map):
    if unified_hash(pair) in setting_map:
        assert pair == setting_map[unified_hash(pair)], "config with same hash but different value"
    else:
        setting_map[unified_hash(pair)] = pair

setting_map = {}

repacking_methods = [RepackMethod.NONE, RepackMethod.GREEDY, RepackMethod.MEDIAN]
pack_sizes = [2, 4, 8, 16, 32]

for model_name in model_list:
    for repacking_method in repacking_methods:
        for pack_size in pack_sizes:
            config = PackKVCacheConfig(
                        enable_quant=True,
                        model_name=model_name,
                        quant_method=QuantMethod.PackKV,
                        repack_method=repacking_method,
                        high_precision_zero_point=False,
                        block_size=BLOCK_SIZE,
                        buffer_size=BUFFER_SIZE,
                        pack_size=pack_size,
                        k_quant_scale_rel=k_rel_quant_scale,
                        v_quant_scale_rel=v_rel_quant_scale
            )
            add_to_map(config, setting_map)

setting_path = "data/pack_size_cr/pack_size_cr_setting_map.pkl"
print(setting_map)
save(setting_map, setting_path)
logger.info(f"Setting map saved to {setting_path}, total {len(setting_map)} entries")