#! /usr/bin/env python
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.compute import QuantMethod, RepackMethod
from utils.config import PackKVCacheConfig
from utils.util import get_logger, block_other_logger
from evaluation.evaluation import accuracy_evaluation, cr_evaluation, throughput_evaluation

logger = get_logger(__file__)
block_other_logger(logger)
# register_notify()

BUFFER_SIZE = 128 + 64
BLOCK_SIZE = 64
PACK_SIZE = 16

config = PackKVCacheConfig(
                    enable_quant=False,
                    model_name="meta-llama/Llama-3.1-8B",
                    quant_method=QuantMethod.PackKV,
                    repack_method=RepackMethod.NONE,
                    high_precision_zero_point=False,
                    block_size=BLOCK_SIZE,
                    buffer_size=BUFFER_SIZE,
                    pack_size=PACK_SIZE,
                    k_quant_scale_rel=0.1,
                    v_quant_scale_rel=0.2,
)

crs = throughput_evaluation(
    config=config,
    ctx_len= 64 * 1024,
    enable_save=False,
    logger=logger
)

layer_num = len(crs[0]["k_original_size"])

k_original_size = 0
v_original_size = 0
k_compressed_size = 0
v_compressed_size = 0
k_our_kernel_time = 0
v_our_kernel_time = 0
k_our_none_kernel_time = 0
v_our_none_kernel_time = 0
k_pytorch_kernel_time = 0
v_pytorch_kernel_time = 0

for layer_idx in range(layer_num):
    k_original_size += crs[0]["k_original_size"][layer_idx]
    v_original_size += crs[0]["v_original_size"][layer_idx]
    k_compressed_size += crs[0]["k_compressed_size"][layer_idx]
    v_compressed_size += crs[0]["v_compressed_size"][layer_idx]
    k_our_kernel_time += crs[0]["k_our_kernel_time"][layer_idx]
    v_our_kernel_time += crs[0]["v_our_kernel_time"][layer_idx]
    k_our_none_kernel_time += crs[0]["k_our_none_kernel_time"][layer_idx]
    v_our_none_kernel_time += crs[0]["v_our_none_kernel_time"][layer_idx]
    k_pytorch_kernel_time += crs[0]["k_pytorch_kernel_time"][layer_idx]
    v_pytorch_kernel_time += crs[0]["v_pytorch_kernel_time"][layer_idx]

# print(crs)
print("k_original_size: ", k_original_size)
print("v_original_size: ", v_original_size)
print("k_compressed_size: ", k_compressed_size)
print("v_compressed_size: ", v_compressed_size)
print("k_our_kernel_time: ", k_our_kernel_time)
print("v_our_kernel_time: ", v_our_kernel_time)
print("k_our_none_kernel_time: ", k_our_none_kernel_time)
print("v_our_none_kernel_time: ", v_our_none_kernel_time)
print("k_pytorch_kernel_time: ", k_pytorch_kernel_time)
print("v_pytorch_kernel_time: ", v_pytorch_kernel_time)
print(crs)

# v_our_kernel_time:  4.145056001842022