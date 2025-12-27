#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import PackKVCacheConfig
from utils.serialization import load, save
from utils.util import get_logger, block_other_logger, register_notify
from utils.compute import QuantMode
from matplotlib.font_manager import FontProperties

font_path = 'Founders_Grotesk/FoundersGrotesk-Regular.otf'
founders_reg_prop = FontProperties(fname=font_path)

def pair(settings, results):
    assert len(settings) == len(results)
    pairs = []
    for setting, result in zip(settings, results):
        pair = (setting, result)
        pairs.append(pair)
    return pairs

def filter(pairs, func):
    new_pairs = []
    for pair in pairs:
        setting, result = pair
        if func(setting, result):
            new_pairs.append(pair)
    return new_pairs

def draw_one_line_pic(x, y, y_label, title, save_path, benchmark):
    x = np.array(x)
    y = np.array(y)

    plt.figure(figsize=(6, 3))
    
    # Plot original data points using actual x values
    plt.plot(x, y, marker='o', color='#356ba0', linewidth=2, label=y_label)
    
    # Add cubic spline interpolation for smoother curve fitting
    if len(x) >= 3:  # Need at least 3 points for fitting
        from scipy.interpolate import CubicSpline
        
        # Use CubicSpline which passes through all data points while maintaining smoothness
        # This provides a good balance between smoothness and fidelity to data points
        spline = CubicSpline(x, y, bc_type='natural')
        
        # Generate smooth curve points
        x_smooth = np.linspace(min(x), max(x), 200)  # More points for smoother appearance
        y_smooth = spline(x_smooth)
        
        # Plot fitted curve
        plt.plot(x_smooth, y_smooth, color='#ff6b6b', linewidth=2, linestyle='-', alpha=0.8, label='Fitted Curve')

    # Add a horizontal dashed line at 0.95 * y[0]
    baseline_value = 0.95 * y[0]
    plt.axhline(y=baseline_value, color='gray', linestyle='--', alpha=0.7)
    
    # Add text label for the baseline
    ax = plt.gca()
    xlim = ax.get_xlim()
    plt.text(xlim[1] * 0.98, baseline_value, f'95% baseline ({baseline_value:.3f})', 
             va='center', ha='right', color='gray', fontsize=10, 
             backgroundcolor='white', alpha=0.8)

    # Add vertical lines for 4bit, 3bit and 2bit quantization scales at specific x-axis positions
    bit4_scale = 1.0 / (16 - 1)  # 4bit: 1/15 ≈ 0.067
    bit3_scale = 1.0 / (8 - 1)   # 3bit: 1/7 ≈ 0.143
    bit2_scale = 1.0 / (4 - 1)   # 2bit: 1/3 ≈ 0.333
    
    # Check if these scale values are within the x-axis range
    x_array = np.array(x)
    if len(x_array) > 0:
        x_min, x_max = min(x_array), max(x_array)
        
        # Add 4bit line if within range
        if x_min <= bit4_scale <= x_max:
            plt.axvline(x=bit4_scale, color='green', linestyle=':', alpha=0.8, linewidth=2)
            ylim = ax.get_ylim()
            # Place 4bit label at bottom to avoid conflicts
            plt.text(bit4_scale, ylim[0] + (ylim[1] - ylim[0]) * 0.15, '4bit', ha='center', va='bottom', 
                    color='green', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add 3bit line if within range
        if x_min <= bit3_scale <= x_max:
            plt.axvline(x=bit3_scale, color='orange', linestyle=':', alpha=0.8, linewidth=2)
            ylim = ax.get_ylim()
            # Place 3bit label at middle height
            plt.text(bit3_scale, ylim[0] + (ylim[1] - ylim[0]) * 0.35, '3bit', ha='center', va='center', 
                    color='orange', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add 2bit line if within range
        if x_min <= bit2_scale <= x_max:
            plt.axvline(x=bit2_scale, color='red', linestyle=':', alpha=0.8, linewidth=2)
            ylim = ax.get_ylim()
            # Place 2bit label at bottom to avoid top conflicts
            plt.text(bit2_scale, ylim[0] + (ylim[1] - ylim[0]) * 0.25, '2bit', ha='center', va='center', 
                    color='red', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # X-axis now uses actual quantization scale values, no need to manually set xticks
    # The scale values will be automatically displayed proportionally
    # # Add a horizontal dashed line at y=0.97
    # plt.axhline(y=0.97, color='gray', linestyle='--', alpha=0.7)
    # ax = plt.gca()
    # xlim = ax.get_xlim()
    # # Position the text near the right edge, vertically centered on the line
    # # Adjust the x-coordinate (e.g., xlim[1] * 0.95) and alignment as needed
    # plt.text(xlim[1] * 0.98, 0.97, '0.97', va='center', ha='right', color='gray', fontsize=12, backgroundcolor='white', alpha=0.8)

    # Apply the specified font and increase font size for labels
    plt.xlabel("Relative Quantization Scale", fontproperties=founders_reg_prop, fontsize=16)
    plt.ylabel(f"{benchmark} Accuracy", fontproperties=founders_reg_prop, fontsize=16)
    # Optionally apply to title as well
    # plt.title(title, fontproperties=founders_reg_prop, fontsize=16)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    # Increase font size for tick labels
    plt.xticks(fontproperties=founders_reg_prop, fontsize=11)
    plt.yticks(fontproperties=founders_reg_prop, fontsize=16)
    plt.legend(prop=founders_reg_prop, fontsize=12)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.savefig(os.path.join(save_path, f"{benchmark}[{title}].pdf"))
    plt.close()

logger = get_logger(__file__)
block_other_logger(logger)
setting_path = "../data/accuracy/accuracy_setting_map.pkl"
setting_map: Dict[int, Tuple[str, PackKVCacheConfig]] = load(setting_path)

save_path = "../data/accuracy/accuracy_result_map.pkl"
accuracy_result_map = load(save_path)

pairs = pair(setting_map.values(), accuracy_result_map.values())

model_list = []
benchmark_list = []
for pair in pairs:
    setting, result = pair
    if setting[1].model_name not in model_list:
        model_list.append(setting[1].model_name)
    if setting[0] not in benchmark_list:
        benchmark_list.append(setting[0])

benchmark_key_map = {
    "coqa": "em,none",
    "gsm8k": "exact_match,strict-match",
    "mmlu": "acc,none",
    "gpqa": "acc,none",
    "winogrande": "acc,none",
    "squad_completion": "contains,none"
}

def extract_scales_accuracies(pairs, benchmark, is_k: bool):
    scales = []
    accuracies = []
    for pair in pairs:
        setting, result = pair
        scales.append(setting[1].k_quant_scale_rel if is_k else setting[1].v_quant_scale_rel)
        if isinstance(benchmark, list):
            accuracy = 0
            for benchmark_ in benchmark:
                accuracy += result[benchmark_][benchmark_key_map["gpqa"]]
            accuracies.append(accuracy / len(benchmark))
        else:
            accuracies.append(result[benchmark][benchmark_key_map[benchmark]])
    return scales, accuracies


for model in model_list:
    for benchmark in benchmark_list:
        model_name_only = model.split('/')[-1] if '/' in model else model

        # K channel quant
        filtered_pairs = filter(pairs, lambda setting, _: setting[1].model_name == model and setting[0] == benchmark and setting[1].quant_method.value[0] == QuantMode.ChannelQuant)
        scales, accuracies = extract_scales_accuracies(filtered_pairs, benchmark, True)
        # Extract just the model name (without organization prefix)
        draw_one_line_pic(scales, accuracies, "K Channel Quant", f"K_Channel_Quant_{model_name_only}",
                          "accuracy_turning_point", "gpqa" if isinstance(benchmark, list) else benchmark)
        # K token quant
        filtered_pairs = filter(pairs, lambda setting, _: setting[1].model_name == model and setting[0] == benchmark and setting[1].quant_method.value[0] == QuantMode.TokenQuant and setting[1].v_quant_scale_rel == 0.01)
        scales, accuracies = extract_scales_accuracies(filtered_pairs, benchmark, True)
        # Extract just the model name (without organization prefix)
        draw_one_line_pic(scales, accuracies, "K Token Quant", f"K_Token_Quant_{model_name_only}",
                          "accuracy_turning_point", "gpqa" if isinstance(benchmark, list) else benchmark)
        # V token quant
        filtered_pairs = filter(pairs, lambda setting, _: setting[1].model_name == model and setting[0] == benchmark and setting[1].quant_method.value[1] == QuantMode.TokenQuant and setting[1].quant_method.value[0] == QuantMode.TokenQuant and setting[1].k_quant_scale_rel == 0.01)
        scales, accuracies = extract_scales_accuracies(filtered_pairs, benchmark, False)
        # Extract just the model name (without organization prefix)
        draw_one_line_pic(scales, accuracies, "V Token Quant", f"V_Token_Quant_{model_name_only}",
                          "accuracy_turning_point", "gpqa" if isinstance(benchmark, list) else benchmark)

# # "meta-llama/Llama-2-13b-hf"
# ## k channel quant
# filtered_pairs = filter(pairs, lambda setting, _: setting[1].quant_method.value[0] == QuantMode.ChannelQuant and setting[1].model_name == "meta-llama/Llama-2-13b-hf")
# enable_high_precision_zero_point, disable_high_precision_zero_point, enable_high_precision_zero_point_quant_scales, disable_high_precision_zero_point_quant_scales = extract_scales_accuracies(filtered_pairs, True)
# assert  enable_high_precision_zero_point_quant_scales == disable_high_precision_zero_point_quant_scales
#
# draw_two_lines_pic(enable_high_precision_zero_point_quant_scales, enable_high_precision_zero_point, disable_high_precision_zero_point, "Enable HP ZP", "Disable HP ZP", "Llama_2_13b_hf_K_Channel_Quant", "./zero_point", "gsm8k")
#
# ## K token quant
# filtered_pairs = filter(pairs, lambda setting, _: setting[1].quant_method.value[0] == QuantMode.TokenQuant and setting[1].v_quant_scale_rel == 0.01 and setting[1].model_name == "meta-llama/Llama-2-13b-hf")
# enable_high_precision_zero_point, disable_high_precision_zero_point, enable_high_precision_zero_point_quant_scales, disable_high_precision_zero_point_quant_scales = extract_scales_accuracies(filtered_pairs, True)
# assert  enable_high_precision_zero_point_quant_scales == disable_high_precision_zero_point_quant_scales
#
# draw_two_lines_pic(enable_high_precision_zero_point_quant_scales, enable_high_precision_zero_point, disable_high_precision_zero_point, "Enable HP ZP", "Disable HP ZP", "Llama_2_13b_hf_K_Token_Quant", "./zero_point", "gsm8k")
#
# ## V token quant
# filtered_pairs = filter(pairs, lambda setting, result: setting[1].quant_method.value[1] == QuantMode.TokenQuant and setting[1].k_quant_scale_rel == 0.01 and setting[1].model_name == "meta-llama/Llama-2-13b-hf")
# enable_high_precision_zero_point, disable_high_precision_zero_point, enable_high_precision_zero_point_quant_scales, disable_high_precision_zero_point_quant_scales = extract_scales_accuracies(filtered_pairs, False)
# assert  enable_high_precision_zero_point_quant_scales == disable_high_precision_zero_point_quant_scales
#
# draw_two_lines_pic(enable_high_precision_zero_point_quant_scales, enable_high_precision_zero_point, disable_high_precision_zero_point, "Enable HP ZP", "Disable HP ZP", "Llama_2_13b_hf_V_Token_Quant", "./zero_point", "gsm8k")
#
# # "Qwen/Qwen3-8B"
# ## K channel quant
# filtered_pairs = filter(pairs, lambda setting, _: setting[1].quant_method.value[0] == QuantMode.ChannelQuant and setting[1].model_name == "Qwen/Qwen3-8B")
# enable_high_precision_zero_point, disable_high_precision_zero_point, enable_high_precision_zero_point_quant_scales, disable_high_precision_zero_point_quant_scales = extract_scales_accuracies(filtered_pairs, True)
# assert  enable_high_precision_zero_point_quant_scales == disable_high_precision_zero_point_quant_scales
#
# draw_two_lines_pic(enable_high_precision_zero_point_quant_scales, enable_high_precision_zero_point, disable_high_precision_zero_point, "Enable HP ZP", "Disable HP ZP", "Qwen3_8B_K_Channel_Quant", "./zero_point", "gsm8k")
#
# ## K token quant
# filtered_pairs = filter(pairs, lambda setting, _: setting[1].quant_method.value[0] == QuantMode.TokenQuant and setting[1].v_quant_scale_rel == 0.01 and setting[1].model_name == "Qwen/Qwen3-8B")
# enable_high_precision_zero_point, disable_high_precision_zero_point, enable_high_precision_zero_point_quant_scales, disable_high_precision_zero_point_quant_scales = extract_scales_accuracies(filtered_pairs, True)
# assert  enable_high_precision_zero_point_quant_scales == disable_high_precision_zero_point_quant_scales
#
# draw_two_lines_pic(enable_high_precision_zero_point_quant_scales, enable_high_precision_zero_point, disable_high_precision_zero_point, "Enable HP ZP", "Disable HP ZP", "Qwen3_8B_K_Token_Quant", "./zero_point", "gsm8k")
#
# ## V token quant
# filtered_pairs = filter(pairs, lambda setting, _: setting[1].quant_method.value[1] == QuantMode.TokenQuant and setting[1].k_quant_scale_rel == 0.01 and setting[1].model_name == "Qwen/Qwen3-8B")
# enable_high_precision_zero_point, disable_high_precision_zero_point, enable_high_precision_zero_point_quant_scales, disable_high_precision_zero_point_quant_scales = extract_scales_accuracies(filtered_pairs, False)
# assert  enable_high_precision_zero_point_quant_scales == disable_high_precision_zero_point_quant_scales
#
# draw_two_lines_pic(enable_high_precision_zero_point_quant_scales, enable_high_precision_zero_point, disable_high_precision_zero_point, "Enable HP ZP", "Disable HP ZP", "Qwen3_8B_V_Token_Quant", "./zero_point", "gsm8k")