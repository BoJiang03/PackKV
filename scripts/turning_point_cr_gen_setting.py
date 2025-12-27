#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.serialization import load, unified_hash, save
from utils.compute import QuantMode, QuantMethod, RepackMethod
from utils.config import PackKVCacheConfig

turning_points_path = "data/turning_points/turning_points.pkl"
turning_points = load(turning_points_path)

print(len(turning_points))

pack_sizes = [4, 8, 16]
repack_methods = [
    RepackMethod.NONE,
    RepackMethod.GREEDY,
    RepackMethod.MEDIAN
]

cr_setting_maps = {}

BLOCK_SIZE = 64
BUFFER_SIZE = 128 + 64

for turning_point in turning_points:
    if turning_point["is_k"] and turning_point['quantization_mode'] == QuantMode.ChannelQuant:
        config = PackKVCacheConfig(
            model_name=turning_point["model"],
            quant_method=QuantMethod.KIVI,
            repack_method=RepackMethod.NONE,
            high_precision_zero_point=False,
            block_size=BLOCK_SIZE,
            buffer_size=BUFFER_SIZE,
            pack_size=16,
            k_quant_scale_rel=turning_point["turning_point"],
            v_quant_scale_rel=0.01,
        )
        cr_setting_maps[unified_hash(config)] = config
        continue

    for pack_size in pack_sizes:
        for repack_method in repack_methods:
            config = PackKVCacheConfig(
                model_name=turning_point["model"],
                quant_method=QuantMethod.PackKV,
                repack_method=repack_method,
                high_precision_zero_point=False,
                block_size=BLOCK_SIZE,
                buffer_size=BUFFER_SIZE,
                pack_size=pack_size,
                k_quant_scale_rel= turning_point["turning_point"] if turning_point["is_k"] else 0.01,
                v_quant_scale_rel= turning_point["turning_point"] if not turning_point["is_k"] else 0.01,
            )
            cr_setting_maps[unified_hash(config)] = config

turning_point_cr_setting_path = "data/turning_point/turning_point_cr_setting_map.pkl"
save(cr_setting_maps, turning_point_cr_setting_path)
print(f"Saved {len(cr_setting_maps)} turning point cr setting")
