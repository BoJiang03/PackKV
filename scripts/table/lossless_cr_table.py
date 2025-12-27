#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os
import numpy as np

from utils.compute import RepackMethod

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import PackKVCacheConfig
from utils.serialization import load, save
from utils.util import get_logger, block_other_logger, register_notify

def pair(settings, results):
    assert len(settings) == len(results)
    pairs = []
    for setting, result in zip(settings, results):
        pair = (setting, result)
        pairs.append(pair)
    return pairs

def filter_paris(pairs, func):
    filtered_pairs = []
    for pair in pairs:
        setting, result = pair
        if func(setting, result):
            filtered_pairs.append(pair)
    return filtered_pairs

def get_visualization_datas(filtered_pairs):
    pack_sizes = [setting.pack_size for setting, _ in filtered_pairs]
    k_original_sizes = [sum(result[0]['k_original_size']) for _, result in filtered_pairs]
    v_original_sizes = [sum(result[0]['v_original_size']) for _, result in filtered_pairs]
    k_quant_sizes = [sum(result[0]['k_quant_size']) for _, result in filtered_pairs]
    v_quant_sizes = [sum(result[0]['v_quant_size']) for _, result in filtered_pairs]
    k_encode_before_repack_sizes = [sum(result[0]['k_encode_size_before_repack']) for _, result in filtered_pairs]
    v_encode_before_repack_sizes = [sum(result[0]['v_encode_size_before_repack']) for _, result in filtered_pairs]
    k_encode_after_repack_sizes = [sum(result[0]['k_encode_size_after_repack']) for _, result in filtered_pairs]
    v_encode_after_repack_sizes = [sum(result[0]['v_encode_size_after_repack']) for _, result in filtered_pairs]

    k_quant_crs = [k_original_size / k_quant_size for k_original_size, k_quant_size in zip(k_original_sizes, k_quant_sizes)]
    v_quant_crs = [v_original_size / v_quant_size for v_original_size, v_quant_size in zip(v_original_sizes, v_quant_sizes)]
    k_encode_before_repack_crs = [k_original_size / k_encode_before_repack_size for k_original_size, k_encode_before_repack_size in zip(k_original_sizes, k_encode_before_repack_sizes)]
    v_encode_before_repack_crs = [v_original_size / v_encode_before_repack_size for v_original_size, v_encode_before_repack_size in zip(v_original_sizes, v_encode_before_repack_sizes)]
    k_encode_after_repack_crs = [k_original_size / k_encode_after_repack_size for k_original_size, k_encode_after_repack_size in zip(k_original_sizes, k_encode_after_repack_sizes)]
    v_encode_after_repack_crs = [v_original_size / v_encode_after_repack_size for v_original_size, v_encode_after_repack_size in zip(v_original_sizes, v_encode_after_repack_sizes)]

    return pack_sizes, k_quant_crs, v_quant_crs, k_encode_before_repack_crs, v_encode_before_repack_crs, k_encode_after_repack_crs, v_encode_after_repack_crs

def get_table_data(pack_sizes, quant_crs, encode_before_repack_crs, greedy_repack_crs, median_repack_crs):
    quant_cr = quant_crs[0]
    max_encode_before_repack_cr = max(encode_before_repack_crs)
    max_encode_before_repack_cr_index = encode_before_repack_crs.index(max_encode_before_repack_cr)
    max_greedy_repack_cr = max(greedy_repack_crs)
    max_greedy_repack_cr_index = greedy_repack_crs.index(max_greedy_repack_cr)
    max_median_repack_cr = max(median_repack_crs)
    max_median_repack_cr_index = median_repack_crs.index(max_median_repack_cr)
    max_crs = [max_encode_before_repack_cr, max_greedy_repack_cr, max_median_repack_cr]
    max_crs_index = [max_encode_before_repack_cr_index, max_greedy_repack_cr_index, max_median_repack_cr_index]
    max_cr_repack_method = [RepackMethod.NONE, RepackMethod.GREEDY, RepackMethod.MEDIAN]
    max_cr = max(max_crs)
    max_cr_index = max_crs.index(max_cr)
    max_cr_repack_method = max_cr_repack_method[max_cr_index]
    max_cr_pack_size_idx = max_crs_index[max_cr_index]
    max_cr_pack_size = pack_sizes[max_cr_pack_size_idx]
    return {
        "max_cr": max_cr,
        "max_cr_pack_size": max_cr_pack_size,
        "max_cr_repack_method": max_cr_repack_method,
        "quant_cr": quant_cr,
        "none_repack_encode_cr": encode_before_repack_crs[max_cr_pack_size_idx],
        "greedy_repack_encode_cr": greedy_repack_crs[max_cr_pack_size_idx],
        "median_repack_encode_cr": median_repack_crs[max_cr_pack_size_idx],
        "repack_improvement": max_cr / encode_before_repack_crs[max_cr_pack_size_idx] - 1
    }

# Main execution
logger = get_logger(__file__)
block_other_logger(logger)

# Load data
setting_path = "../data/lossless_cr/lossless_setting_map.pkl"
setting_map: Dict[int, PackKVCacheConfig] = load(setting_path)

result_path = "../data/lossless_cr/lossless_result_map.pkl"
accuracy_result_map = load(result_path)

pairs = pair(setting_map.values(), accuracy_result_map.values())

model_list = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-2-13b-hf",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "mistralai/Ministral-8B-Instruct-2410",
    "microsoft/phi-4"
]

# Process data for all models
table_datas = []

for model_name in model_list:
    filtered_pairs = filter_paris(pairs, lambda setting, result: setting.model_name == model_name and setting.repack_method == RepackMethod.NONE)
    pack_sizes, k_quant_crs, v_quant_crs, k_encode_before_repack_crs, v_encode_before_repack_crs, _, _ = get_visualization_datas(filtered_pairs)
    filtered_pairs = filter_paris(pairs, lambda setting, result: setting.model_name == model_name and setting.repack_method == RepackMethod.MEDIAN)
    _, _, _, _, _, k_median_repack_crs, v_median_repack_crs = get_visualization_datas(filtered_pairs)
    filtered_pairs = filter_paris(pairs, lambda setting, result: setting.model_name == model_name and setting.repack_method == RepackMethod.GREEDY)
    _, _, _, _, _, k_greedy_repack_crs, v_greedy_repack_crs = get_visualization_datas(filtered_pairs)
    
    k_table_data = get_table_data(pack_sizes, k_quant_crs, k_encode_before_repack_crs, k_greedy_repack_crs, k_median_repack_crs)
    v_table_data = get_table_data(pack_sizes, v_quant_crs, v_encode_before_repack_crs, v_greedy_repack_crs, v_median_repack_crs)
    table_data = {
        "model_name": model_name,
        "k": k_table_data,
        "v": v_table_data
    }
    table_datas.append(table_data)

# Create model name mapping
model_name_map = {
    "meta-llama/Llama-2-7b-hf": "Llama-2-7B",
    "meta-llama/Llama-3.1-8B": "Llama-3.1-8B", 
    "meta-llama/Llama-2-13b-hf": "Llama-2-13B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "R1-Llama-8B",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral-8B",
    "microsoft/phi-4": "Phi-4"
}

# Organize data by model for column-based table
model_data = {}
for data in table_datas:
    model_name = model_name_map.get(data["model_name"], data["model_name"].split("/")[-1])
    k_data = data["k"]
    v_data = data["v"]
    
    # Calculate percentage improvements
    k_greedy_improvement = ((k_data["greedy_repack_encode_cr"] / k_data["none_repack_encode_cr"]) - 1) * 100
    k_median_improvement = ((k_data["median_repack_encode_cr"] / k_data["none_repack_encode_cr"]) - 1) * 100
    v_greedy_improvement = ((v_data["greedy_repack_encode_cr"] / v_data["none_repack_encode_cr"]) - 1) * 100
    v_median_improvement = ((v_data["median_repack_encode_cr"] / v_data["none_repack_encode_cr"]) - 1) * 100
    
    # Format improvement percentages with correct sign
    k_greedy_sign = "+" if k_greedy_improvement >= 0 else ""
    k_median_sign = "+" if k_median_improvement >= 0 else ""
    v_greedy_sign = "+" if v_greedy_improvement >= 0 else ""
    v_median_sign = "+" if v_median_improvement >= 0 else ""
    
    # Determine which values to highlight for K
    k_none_val = f"{k_data['none_repack_encode_cr']:.2f}"
    k_greedy_val = f"{k_data['greedy_repack_encode_cr']:.2f} ({k_greedy_sign}{k_greedy_improvement:.1f}\\%)"
    k_median_val = f"{k_data['median_repack_encode_cr']:.2f} ({k_median_sign}{k_median_improvement:.1f}\\%)"
    
    if k_data["max_cr_repack_method"].name == "NONE":
        k_none_val = f"\\cdg{{{k_none_val}}}"
    elif k_data["max_cr_repack_method"].name == "GREEDY":
        k_greedy_val = f"\\cdg{{{k_greedy_val}}}"
    elif k_data["max_cr_repack_method"].name == "MEDIAN":
        k_median_val = f"\\cdg{{{k_median_val}}}"
    
    # Determine which values to highlight for V
    v_none_val = f"{v_data['none_repack_encode_cr']:.2f}"
    v_greedy_val = f"{v_data['greedy_repack_encode_cr']:.2f} ({v_greedy_sign}{v_greedy_improvement:.1f}\\%)"
    v_median_val = f"{v_data['median_repack_encode_cr']:.2f} ({v_median_sign}{v_median_improvement:.1f}\\%)"
    
    if v_data["max_cr_repack_method"].name == "NONE":
        v_none_val = f"\\cdg{{{v_none_val}}}"
    elif v_data["max_cr_repack_method"].name == "GREEDY":
        v_greedy_val = f"\\cdg{{{v_greedy_val}}}"
    elif v_data["max_cr_repack_method"].name == "MEDIAN":
        v_median_val = f"\\cdg{{{v_median_val}}}"
    
    model_data[model_name] = {
        'k_none': k_none_val,
        'k_greedy': k_greedy_val,
        'k_median': k_median_val,
        'v_none': v_none_val,
        'v_greedy': v_greedy_val,
        'v_median': v_median_val
    }

# Calculate improvement averages
k_greedy_improvements = []
k_median_improvements = []
v_greedy_improvements = []
v_median_improvements = []

for data in table_datas:
    k_data = data["k"]
    v_data = data["v"]
    
    k_greedy_improvement = ((k_data["greedy_repack_encode_cr"] / k_data["none_repack_encode_cr"]) - 1) * 100
    k_median_improvement = ((k_data["median_repack_encode_cr"] / k_data["none_repack_encode_cr"]) - 1) * 100
    v_greedy_improvement = ((v_data["greedy_repack_encode_cr"] / v_data["none_repack_encode_cr"]) - 1) * 100
    v_median_improvement = ((v_data["median_repack_encode_cr"] / v_data["none_repack_encode_cr"]) - 1) * 100
    
    k_greedy_improvements.append(k_greedy_improvement)
    k_median_improvements.append(k_median_improvement)
    v_greedy_improvements.append(v_greedy_improvement)
    v_median_improvements.append(v_median_improvement)

k_greedy_avg = sum(k_greedy_improvements) / len(k_greedy_improvements)
k_median_avg = sum(k_median_improvements) / len(k_median_improvements)
v_greedy_avg = sum(v_greedy_improvements) / len(v_greedy_improvements)
v_median_avg = sum(v_median_improvements) / len(v_median_improvements)

# Format average improvements
k_greedy_avg_str = f"{k_greedy_avg:+.1f}\\%"
k_median_avg_str = f"{k_median_avg:+.1f}\\%"
v_greedy_avg_str = f"{v_greedy_avg:+.1f}\\%"
v_median_avg_str = f"{v_median_avg:+.1f}\\%"

# Print in LaTeX table format (column-based)
print("\n% LaTeX Table Format (Column-based):")
print("\\begin{table*}[htbp]")
print("\\centering")
print("\\begin{tabular}{clccccccc}")
print("\\toprule")

# Print header
model_names = list(model_data.keys())
header = "\\textbf{Cache} & \\textbf{Mode} & " + " & ".join([f"\\textbf{{{name}}}" for name in model_names]) + " & \\textbf{Avg} \\\\"
print(header)
print("\\midrule")

# Print K rows
print("\\multirow{3}{*}{K}")
print("    & None   & " + " & ".join([model_data[name]['k_none'] for name in model_names]) + " & - \\\\")
print("    & Greedy & " + " & ".join([model_data[name]['k_greedy'] for name in model_names]) + " & " + k_greedy_avg_str + " \\\\")
print("    & Median & " + " & ".join([model_data[name]['k_median'] for name in model_names]) + " & " + k_median_avg_str + " \\\\")
print("\\midrule")

# Print V rows
print("\\multirow{3}{*}{V}")
print("    & None   & " + " & ".join([model_data[name]['v_none'] for name in model_names]) + " & - \\\\")
print("    & Greedy & " + " & ".join([model_data[name]['v_greedy'] for name in model_names]) + " & " + v_greedy_avg_str + " \\\\")
print("    & Median & " + " & ".join([model_data[name]['v_median'] for name in model_names]) + " & " + v_median_avg_str + " \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Compression ratios and related metrics for different models, grouped by cache type (K/V).}")
print("\\label{tab:repacking_cr}")
print("\\end{table*}")

# Calculate repacking and bit packing contributions
print("\n% Contribution Analysis:")
k_repacking_contributions = []
v_repacking_contributions = []
k_bit_packing_contributions = []
v_bit_packing_contributions = []

for data in table_datas:
    model_name = model_name_map.get(data["model_name"], data["model_name"].split("/")[-1])
    k_data = data["k"]
    v_data = data["v"]
    
    # K cache contributions
    k_max_cr = k_data["max_cr"]
    k_none_cr = k_data["none_repack_encode_cr"]
    k_quant_cr = k_data["quant_cr"]
    
    if k_max_cr > k_quant_cr:  # Avoid division by zero
        k_repacking_contribution = (k_max_cr - k_none_cr) / (k_max_cr - k_quant_cr)
        k_bit_packing_contribution = 1 - k_repacking_contribution
    else:
        k_repacking_contribution = 0
        k_bit_packing_contribution = 1
    
    k_repacking_contributions.append(k_repacking_contribution)
    k_bit_packing_contributions.append(k_bit_packing_contribution)
    
    # V cache contributions  
    v_max_cr = v_data["max_cr"]
    v_none_cr = v_data["none_repack_encode_cr"]
    v_quant_cr = v_data["quant_cr"]
    
    if v_max_cr > v_quant_cr:  # Avoid division by zero
        v_repacking_contribution = (v_max_cr - v_none_cr) / (v_max_cr - v_quant_cr)
        v_bit_packing_contribution = 1 - v_repacking_contribution
    else:
        v_repacking_contribution = 0
        v_bit_packing_contribution = 1
    
    v_repacking_contributions.append(v_repacking_contribution)
    v_bit_packing_contributions.append(v_bit_packing_contribution)
    
    print(f"{model_name}:")
    print(f"  K - Repacking: {k_repacking_contribution:.3f}, Bit Packing: {k_bit_packing_contribution:.3f}")
    print(f"  V - Repacking: {v_repacking_contribution:.3f}, Bit Packing: {v_bit_packing_contribution:.3f}")

# Calculate averages
k_avg_repacking = sum(k_repacking_contributions) / len(k_repacking_contributions)
k_avg_bit_packing = sum(k_bit_packing_contributions) / len(k_bit_packing_contributions)
v_avg_repacking = sum(v_repacking_contributions) / len(v_repacking_contributions)
v_avg_bit_packing = sum(v_bit_packing_contributions) / len(v_bit_packing_contributions)

print(f"\nAverage Contributions:")
print(f"K Cache - Repacking: {k_avg_repacking:.3f}, Bit Packing: {k_avg_bit_packing:.3f}")
print(f"V Cache - Repacking: {v_avg_repacking:.3f}, Bit Packing: {v_avg_bit_packing:.3f}")

# Overall average
overall_avg_repacking = (k_avg_repacking + v_avg_repacking) / 2
overall_avg_bit_packing = (k_avg_bit_packing + v_avg_bit_packing) / 2

print(f"\nOverall Average:")
print(f"Repacking Contribution: {overall_avg_repacking:.3f}")
print(f"Bit Packing Contribution: {overall_avg_bit_packing:.3f}") 