#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.compute import RepackMethod

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import PackKVCacheConfig
from utils.serialization import load, save
from utils.util import get_logger, block_other_logger, register_notify

from matplotlib.font_manager import FontProperties

font_path = './Founders_Grotesk/FoundersGrotesk-Regular.otf'
founders_reg_prop = FontProperties(fname=font_path)

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

def draw_lines(pack_sizes, quant_cr, before_repack_crs, median_repack_crs, greedy_repack_crs, model_name, title):
    pack_sizes = np.array(pack_sizes)
    before_repack_crs = np.array(before_repack_crs)
    median_repack_crs = np.array(median_repack_crs)
    greedy_repack_crs = np.array(greedy_repack_crs)

    plt.figure(figsize=(6, 3))
    
    # Draw the horizontal dashed line for quant_cr
    plt.axhline(y=quant_cr, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Quant CR')
    
    # Add text label for the quant_cr line
    ax = plt.gca()
    xlim = ax.get_xlim()
    plt.text(xlim[1] * 0.98, quant_cr, f'{quant_cr:.3f}', 
             va='center', ha='right', color='gray', fontsize=10, 
             backgroundcolor='white', alpha=0.8)
    
    # Draw the three repack method lines
    plt.plot(range(len(pack_sizes)), before_repack_crs, marker='o', color='#356ba0', linewidth=2, label='Before Repack')
    plt.plot(range(len(pack_sizes)), median_repack_crs, marker='s', color='#d62728', linewidth=2, label='Median Repack')
    plt.plot(range(len(pack_sizes)), greedy_repack_crs, marker='^', color='#2ca02c', linewidth=2, label='Greedy Repack')

    # Set x-axis labels
    plt.xticks(range(len(pack_sizes)), [f'{t:.0f}' if isinstance(t, float) else str(t) for t in pack_sizes])
    
    # Apply the specified font and increase font size for labels
    plt.xlabel("Pack Size", fontproperties=founders_reg_prop, fontsize=16)
    plt.ylabel(f"Compression Ratio", fontproperties=founders_reg_prop, fontsize=16)
    
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    # Increase font size for tick labels
    plt.xticks(fontproperties=founders_reg_prop, fontsize=11)
    plt.yticks(fontproperties=founders_reg_prop, fontsize=16)
    plt.legend(prop=founders_reg_prop, fontsize=12)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    
    # Clean model name for filename
    clean_model_name = model_name.split("/")[-1].replace("-", "_")
    plt.savefig(os.path.join(save_path, f"{clean_model_name}_{title}.pdf"))
    plt.close()

def create_pie_chart(repacking_contrib, bit_packing_contrib, title, filename):
    labels = ['Repacking', 'Bit Packing']
    sizes = [repacking_contrib, bit_packing_contrib]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0)  # explode the repacking slice slightly
    
    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=None, colors=colors, autopct='%1.1f%%',
                                       shadow=True, startangle=90, textprops={'fontsize': 20})
    
    # Apply custom font to percentages
    for autotext in autotexts:
        autotext.set_fontproperties(founders_reg_prop)
        autotext.set_fontsize(22)
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    # Add legend in the upper right corner
    legend_font = FontProperties(fname=font_path, size=24)
    plt.legend(wedges, labels, loc='upper right', prop=legend_font)
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()

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

save_path = "lossless_cr_contribution/"

# Process data for all models
table_datas = []

for model_name in model_list:
    filtered_pairs = filter_paris(pairs, lambda setting, result: setting.model_name == model_name and setting.repack_method == RepackMethod.NONE)
    pack_sizes, k_quant_crs, v_quant_crs, k_encode_before_repack_crs, v_encode_before_repack_crs, _, _ = get_visualization_datas(filtered_pairs)
    filtered_pairs = filter_paris(pairs, lambda setting, result: setting.model_name == model_name and setting.repack_method == RepackMethod.MEDIAN)
    _, _, _, _, _, k_median_repack_crs, v_median_repack_crs = get_visualization_datas(filtered_pairs)
    filtered_pairs = filter_paris(pairs, lambda setting, result: setting.model_name == model_name and setting.repack_method == RepackMethod.GREEDY)
    _, _, _, _, _, k_greedy_repack_crs, v_greedy_repack_crs = get_visualization_datas(filtered_pairs)
    
    # Generate line plots (uncomment if needed)
    # draw_lines(pack_sizes, k_quant_crs[0], k_encode_before_repack_crs, k_median_repack_crs, k_greedy_repack_crs, model_name, "k")
    # draw_lines(pack_sizes, v_quant_crs[0], v_encode_before_repack_crs, v_median_repack_crs, v_greedy_repack_crs, model_name, "v")
    
    k_table_data = get_table_data(pack_sizes, k_quant_crs, k_encode_before_repack_crs, k_greedy_repack_crs, k_median_repack_crs)
    v_table_data = get_table_data(pack_sizes, v_quant_crs, v_encode_before_repack_crs, v_greedy_repack_crs, v_median_repack_crs)
    table_data = {
        "model_name": model_name,
        "k": k_table_data,
        "v": v_table_data
    }
    table_datas.append(table_data)

# Calculate repacking and bit packing contributions for pie charts
print("Calculating contribution analysis for pie charts...")
k_repacking_contributions = []
v_repacking_contributions = []
k_bit_packing_contributions = []
v_bit_packing_contributions = []

for data in table_datas:
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

# Calculate averages
k_avg_repacking = sum(k_repacking_contributions) / len(k_repacking_contributions)
k_avg_bit_packing = sum(k_bit_packing_contributions) / len(k_bit_packing_contributions)
v_avg_repacking = sum(v_repacking_contributions) / len(v_repacking_contributions)
v_avg_bit_packing = sum(v_bit_packing_contributions) / len(v_bit_packing_contributions)

print(f"Average Contributions:")
print(f"K Cache - Repacking: {k_avg_repacking:.3f}, Bit Packing: {k_avg_bit_packing:.3f}")
print(f"V Cache - Repacking: {v_avg_repacking:.3f}, Bit Packing: {v_avg_bit_packing:.3f}")

# Create pie charts
create_pie_chart(k_avg_repacking, k_avg_bit_packing, 
                'K Cache Compression Ratio Contribution', 'k_cache_contribution.pdf')

create_pie_chart(v_avg_repacking, v_avg_bit_packing,
                'V Cache Compression Ratio Contribution', 'v_cache_contribution.pdf')

print(f"\nPie charts saved:")
print(f"K Cache: {save_path}k_cache_contribution.pdf")
print(f"V Cache: {save_path}v_cache_contribution.pdf")