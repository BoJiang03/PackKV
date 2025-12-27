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

font_path = '../figure/Founders_Grotesk/FoundersGrotesk-Regular.otf'
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

def draw_two_lines(pack_sizes, k_quant_cr, v_quant_cr, k_optimal_encode_crs, v_optimal_encode_crs, model_name, save_path):
    pack_sizes = np.array(pack_sizes)
    k_optimal_encode_crs = np.array(k_optimal_encode_crs)
    v_optimal_encode_crs = np.array(v_optimal_encode_crs)

    plt.figure(figsize=(6, 3))

    # Draw the horizontal dashed line for quant_cr
    plt.axhline(y=k_quant_cr, color='#356ba0', linestyle='--', alpha=0.7, linewidth=2, label='K Quant CR')
    plt.axhline(y=v_quant_cr, color='#d62728', linestyle='--', alpha=0.7, linewidth=2, label='V Quant CR')

    # Add text label for the quant_cr line
    ax = plt.gca()
    xlim = ax.get_xlim()
    plt.text(xlim[1] * 0.98, k_quant_cr, f'{k_quant_cr:.3f}',
             va='center', ha='right', color='gray', fontsize=10,
             backgroundcolor='white', alpha=0.8)
    plt.text(xlim[1] * 0.98, v_quant_cr, f'{v_quant_cr:.3f}',
             va='center', ha='right', color='gray', fontsize=10,
             backgroundcolor='white', alpha=0.8)

    # Draw the three repack method lines
    plt.plot(range(len(pack_sizes)), k_optimal_encode_crs, marker='o', color='#356ba0', linewidth=2, label='K Encoded CR')
    plt.plot(range(len(pack_sizes)), v_optimal_encode_crs, marker='s', color='#d62728', linewidth=2, label='V Encoded CR')

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
    plt.tight_layout()  # Adjust layout to prevent labels overlapping

    # Clean model name for filename
    clean_model_name = model_name.split("/")[-1].replace("-", "_")
    plt.savefig(os.path.join(save_path, f"{clean_model_name}.pdf"))
    plt.close()

def draw_all_models(all_model_data, cache_type, save_path):
    """
    Draw a single plot with all models' curves for K or V cache
    
    Args:
        all_model_data: List of tuples (model_name, pack_sizes, quant_cr, optimal_encode_crs)
        cache_type: 'K' or 'V'
        save_path: Path to save the figure
    """
    plt.figure(figsize=(8, 4))
    
    # Define colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # Draw horizontal dashed line for quant_cr (use first model's quant_cr, assuming all are the same)
    quant_cr = all_model_data[0][2]
    plt.axhline(y=quant_cr, color='gray', linestyle='--', 
               alpha=0.7, linewidth=2, label=f'{cache_type} Quant CR')
    
    for idx, (model_name, pack_sizes, _, optimal_encode_crs) in enumerate(all_model_data):
        # Get simple model name for legend
        simple_name = model_name.split("/")[-1]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Draw the curve
        plt.plot(range(len(pack_sizes)), optimal_encode_crs, 
                marker=marker, color=color, linewidth=2, 
                label=simple_name, alpha=0.8)
    
    # Set x-axis labels (assuming all models have same pack_sizes)
    pack_sizes = all_model_data[0][1]
    plt.xticks(range(len(pack_sizes)), [f'{t:.0f}' if isinstance(t, float) else str(t) for t in pack_sizes])
    
    # Apply the specified font and increase font size for labels
    plt.xlabel("Pack Size", fontproperties=founders_reg_prop, fontsize=24)
    plt.ylabel(f"{cache_type} Compression Ratio", fontproperties=founders_reg_prop, fontsize=24)
    
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    plt.xticks(fontproperties=founders_reg_prop, fontsize=20)
    plt.yticks(fontproperties=founders_reg_prop, fontsize=24)
    
    # Create FontProperties for legend with larger size
    legend_font = FontProperties(fname=font_path, size=15)
    plt.legend(prop=legend_font, loc='lower right', framealpha=0.2)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, f"all_models_{cache_type.lower()}.pdf"))
    plt.close()

def main():
    logger = get_logger(__file__)
    block_other_logger(logger)
    
    # Load data
    setting_path = "../data/lossless_cr/lossless_setting_map.pkl"
    setting_map: Dict[int, PackKVCacheConfig] = load(setting_path)

    save_path_data = "../data/lossless_cr/lossless_result_map.pkl"
    accuracy_result_map = load(save_path_data)

    pairs = pair(setting_map.values(), accuracy_result_map.values())

    model_list = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-2-13b-hf",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "mistralai/Ministral-8B-Instruct-2410",
        "microsoft/phi-4"
    ]

    save_path = "pack_size_cr/"
    
    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Store data for all models to generate combined plots
    all_k_data = []
    all_v_data = []

    # Generate plots for each model
    for model_name in model_list:
        print(f"Generating plot for {model_name}...")
        
        # Get data for different repack methods
        filtered_pairs = filter_paris(pairs, lambda setting, result: setting.model_name == model_name and setting.repack_method == RepackMethod.NONE)
        pack_sizes, k_quant_crs, v_quant_crs, k_encode_before_repack_crs, v_encode_before_repack_crs, _, _ = get_visualization_datas(filtered_pairs)
        
        filtered_pairs = filter_paris(pairs, lambda setting, result: setting.model_name == model_name and setting.repack_method == RepackMethod.MEDIAN)
        _, _, _, _, _, k_median_repack_crs, v_median_repack_crs = get_visualization_datas(filtered_pairs)
        
        filtered_pairs = filter_paris(pairs, lambda setting, result: setting.model_name == model_name and setting.repack_method == RepackMethod.GREEDY)
        _, _, _, _, _, k_greedy_repack_crs, v_greedy_repack_crs = get_visualization_datas(filtered_pairs)
        
        # Calculate optimal compression ratios
        k_optimal_encode_crs = [max(k_encode_before_repack_crs[i], k_greedy_repack_crs[i], k_median_repack_crs[i]) for i in range(len(pack_sizes))]
        v_optimal_encode_crs = [max(v_encode_before_repack_crs[i], v_greedy_repack_crs[i], v_median_repack_crs[i]) for i in range(len(pack_sizes))]

        # Generate plot for individual model
        draw_two_lines(pack_sizes, k_quant_crs[0], v_quant_crs[0], k_optimal_encode_crs, v_optimal_encode_crs, model_name, save_path)
        
        # Store data for combined plots
        all_k_data.append((model_name, pack_sizes, k_quant_crs[0], k_optimal_encode_crs))
        all_v_data.append((model_name, pack_sizes, v_quant_crs[0], v_optimal_encode_crs))

    # Generate combined plots for all models
    print("\nGenerating combined plot for all models (K cache)...")
    draw_all_models(all_k_data, 'K', save_path)
    
    print("Generating combined plot for all models (V cache)...")
    draw_all_models(all_v_data, 'V', save_path)

    print(f"\nAll plots have been saved to {save_path}")
    print(f"- Individual model plots: {len(model_list)} files")
    print(f"- Combined plots: all_models_k.pdf and all_models_v.pdf")

if __name__ == "__main__":
    main() 