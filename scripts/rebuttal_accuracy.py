#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os

from utils.compute import QuantMethod, RepackMethod
from utils.util import get_logger, block_other_logger

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import accuracy_evaluation
from utils.config import PackKVCacheConfig
logger = get_logger(__file__)
block_other_logger(logger)

benchmarks = [
    # "coqa",
    # "gsm8k",
    # "mmlu",
    # "winogrande",
    # "squad_completion",
    [
        "gpqa_diamond_zeroshot",
        "gpqa_diamond_n_shot",
        # "gpqa_diamond_generative_n_shot",
        # "gpqa_diamond_cot_zeroshot",
        # "gpqa_diamond_cot_n_shot"
    ]
]

model_name = "meta-llama/Llama-2-13b-hf"

BLOCK_SIZE = 64
BUFFER_SIZE = 128 + 64

accuracy_results = {}

benchmark_key_map = {
    "coqa": "em,none",
    "gsm8k": "exact_match,strict-match",
    "mmlu": "acc,none",
    "winogrande": "acc,none",
    "squad_completion": "contains,none",
    "gpqa": "acc,none",
}

for benchmark in benchmarks:
    config = PackKVCacheConfig(
        enable_quant=True,
        model_name=model_name,
        quant_method=QuantMethod.PackKV,
        repack_method=RepackMethod.NONE,
        high_precision_zero_point=False,
        block_size=BLOCK_SIZE,
        buffer_size=BUFFER_SIZE,
        pack_size=16,
        k_quant_scale_rel=0.011,
        v_quant_scale_rel=0.01
    )

    accuracy = accuracy_evaluation(config, benchmark, logger)
    accuracy_error = 0
    if isinstance(benchmark, list):
        for benchmark_ in benchmark:
            accuracy_error += accuracy[benchmark_][benchmark_key_map["gpqa"]]
    else:
        accuracy_error = accuracy[benchmark][benchmark_key_map[benchmark]]

    config = PackKVCacheConfig(
        enable_quant=False,
        model_name=model_name,
        quant_method=QuantMethod.PackKV,
        repack_method=RepackMethod.NONE,
        high_precision_zero_point=False,
        block_size=BLOCK_SIZE,
        buffer_size=BUFFER_SIZE,
        pack_size=16,
        k_quant_scale_rel=0.01,
        v_quant_scale_rel=0.01
    )

    accuracy = accuracy_evaluation(config, benchmark, logger)
    accuracy_no_error = 0
    if isinstance(benchmark, list):
        for benchmark_ in benchmark:
            accuracy_no_error += accuracy[benchmark_][benchmark_key_map["gpqa"]]
    else:
        accuracy_no_error = accuracy[benchmark][benchmark_key_map[benchmark]]

    if isinstance(benchmark, list):
        accuracy_results["gpqa"] = (accuracy_error, accuracy_no_error)
        print(f"gpqa: {accuracy_error}, {accuracy_no_error}")

    else:
        accuracy_results[benchmark] = (accuracy_error, accuracy_no_error)
        print(f"{benchmark}: {accuracy_error}, {accuracy_no_error}")

print(accuracy_results)
