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

no_extend_benchmarks = [
    "coqa",
    "gsm8k",
    "winogrande",
    "squadv2",
    "squad_completion"
]

benchmarks = [
    "coqa",
    "gsm8k",
    "mmlu",
    "winogrande",
    "squad_completion",
    [
        "gpqa_diamond_zeroshot",
        "gpqa_diamond_n_shot",
        # "gpqa_diamond_generative_n_shot",
        # "gpqa_diamond_cot_zeroshot",
        # "gpqa_diamond_cot_n_shot"
    ]
]

channel_quant_k = (0.2, 0.4)
channel_quant_k_ext = (0.2, 0.8)
token_quant_k = (0.06, 0.12)
token_quant_k_ext = (0.06, 0.24)
token_quant_v = (0.12, 0.3)
token_quant_v_ext = (0.12, 0.68)

def gen_quant_rels(_from, _to, part_num):
    quant_rels = []
    part_wide = (_to - _from) / part_num
    for idx in range(part_num + 1):
        quant_rels.append(round(_from + idx * part_wide, 3))
    return quant_rels

def add_to_map(pair, setting_map):
    if unified_hash(pair) in setting_map:
        assert pair == setting_map[unified_hash(pair)], "config with same hash but different value"
    else:
        setting_map[unified_hash(pair)] = pair

channel_quant_k_quant_scale_rels = [0.01] + gen_quant_rels(channel_quant_k[0], channel_quant_k[1], 6)
channel_quant_k_ext_quant_scale_rels = [0.01] + gen_quant_rels(channel_quant_k_ext[0], channel_quant_k_ext[1], 18)
token_quant_k_quant_scale_rels = [0.01] + gen_quant_rels(token_quant_k[0], token_quant_k[1], 6)
token_quant_k_ext_quant_scale_rels = [0.01] + gen_quant_rels(token_quant_k_ext[0], token_quant_k_ext[1], 18)
token_quant_v_quant_scale_rels = [0.01] + gen_quant_rels(token_quant_v[0], token_quant_v[1], 6)
token_quant_v_ext_quant_scale_rels = [0.01] + gen_quant_rels(token_quant_v_ext[0], token_quant_v_ext[1], 18)

setting_map = {}

print(channel_quant_k_quant_scale_rels)
print(channel_quant_k_ext_quant_scale_rels)
print(token_quant_k_quant_scale_rels)
print(token_quant_k_ext_quant_scale_rels)
print(token_quant_v_quant_scale_rels)
print(token_quant_v_ext_quant_scale_rels)

for model_name in model_list:
    for benchmark in benchmarks:
        for quant_scale in (channel_quant_k_ext_quant_scale_rels if benchmark not in no_extend_benchmarks else channel_quant_k_quant_scale_rels):
        # for quant_scale in channel_quant_k_quant_scale_rels:
            # k channel quant
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
            add_to_map((benchmark, config), setting_map)
        for quant_scale in (token_quant_k_ext_quant_scale_rels if benchmark not in no_extend_benchmarks else token_quant_k_quant_scale_rels):
        # for quant_scale in token_quant_k_quant_scale_rels:
            # k token quant
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
            add_to_map((benchmark, config), setting_map)
        for quant_scale in (token_quant_v_ext_quant_scale_rels if benchmark not in no_extend_benchmarks else token_quant_v_quant_scale_rels):
        # for quant_scale in token_quant_v_quant_scale_rels:
            # v token quant
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
            add_to_map((benchmark, config), setting_map)

setting_path = "data/accuracy/accuracy_setting_map.pkl"
print(setting_map)
save(setting_map, setting_path)
logger.info(f"Setting map saved to {setting_path}, total {len(setting_map)} entries")