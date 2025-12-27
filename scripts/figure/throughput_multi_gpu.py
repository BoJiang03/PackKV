#! /usr/bin/env python
from pprint import pprint
from typing import Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import PackKVCacheConfig
from utils.serialization import load, save
from utils.util import get_logger, block_other_logger
import matplotlib.pyplot as plt
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

logger = get_logger(__file__)
block_other_logger(logger)

gpu_nums = 4

gpu_num_throughput_map = {}

for gpu_num in range(1, gpu_nums+1):
    res_ = load(f"../data/throughput_multi_gpu_scaling/throughput_result_map_a100_gpu_{gpu_num}.pkl")
    k_ori_sizes = sum([
        sum(v[0]['k_original_size']) for k, v in res_.items()
    ]) # byte
    v_ori_sizes = sum([
        sum(v[0]['v_original_size']) for k, v in res_.items()
    ]) # byte
    k_kernel_times = sum([
        sum(v[0]['k_our_kernel_time']) + sum(v[0]['k_our_none_kernel_time']) for k, v in res_.items()
    ]) # ms
    v_kernel_times = sum([
        sum(v[0]['v_our_kernel_time']) + sum(v[0]['v_our_none_kernel_time']) for k, v in res_.items()
    ]) # ms
    gpu_num_throughput_map[gpu_num] = {
        "gpu_num": gpu_num,
        "k_throughput": (k_ori_sizes / 1024**3) / (k_kernel_times / 1000),
        "v_throughput": (v_ori_sizes / 1024**3) / (v_kernel_times / 1000),
    }

def draw_scaling_efficiency(gpu_nums, k_throughputs, v_throughputs, save_filename):
    """
    Draw scaling efficiency plot showing how well the system scales with multiple GPUs
    """
    plt.figure(figsize=(8, 5))
    
    # Calculate scaling efficiency
    single_gpu_k = k_throughputs[0]
    single_gpu_v = v_throughputs[0]
    single_gpu_total = single_gpu_k + single_gpu_v
    
    efficiencies = []
    
    for i, gpu_num in enumerate(gpu_nums):
        actual_total_throughput = (k_throughputs[i] + v_throughputs[i]) * gpu_num
        ideal_total_throughput = single_gpu_total * gpu_num
        efficiency = (actual_total_throughput / ideal_total_throughput) * 100
        efficiencies.append(efficiency)
    
    # Plot Scaling Efficiency
    plt.plot(gpu_nums, efficiencies, marker='o', color='#2E8B57', linewidth=3, markersize=8)
    plt.axhline(y=100, color='#FF6B6B', linestyle='--', alpha=0.7, label='Ideal (100%)')
    plt.xlabel("GPU Count", fontproperties=founders_reg_prop, fontsize=32)
    plt.ylabel("Scaling Efficiency (%)", fontproperties=founders_reg_prop, fontsize=32)
    plt.title("Multi-GPU Scaling Efficiency", fontproperties=founders_reg_prop, fontsize=28)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    plt.xticks([1, 2, 3, 4], fontproperties=founders_reg_prop, fontsize=33)
    plt.yticks(fontproperties=founders_reg_prop, fontsize=32)
    plt.ylim(95, 101)
    for i, (gpu_num, eff) in enumerate(zip(gpu_nums, efficiencies)):
        plt.annotate(f'{eff:.1f}%', (gpu_num, eff), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontproperties=founders_reg_prop, fontsize=19)
    
    # Create legend with larger font
    legend = plt.legend(prop=founders_reg_prop, fontsize=19)
    for text in legend.get_texts():
        text.set_fontproperties(founders_reg_prop)
        text.set_fontsize(19)
    plt.tight_layout()
    
    # check if the directory exists
    dir_path = os.path.dirname(save_filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(f"{save_filename}.pdf")
    plt.close()

def draw_total_throughput(gpu_nums, k_throughputs, v_throughputs, save_filename):
    """
    Draw total system throughput vs ideal linear scaling
    """
    plt.figure(figsize=(8, 5))
    
    # Calculate total throughputs
    single_gpu_k = k_throughputs[0]
    single_gpu_v = v_throughputs[0]
    single_gpu_total = single_gpu_k + single_gpu_v
    
    total_throughputs = []
    for i, gpu_num in enumerate(gpu_nums):
        actual_total_throughput = (k_throughputs[i] + v_throughputs[i]) * gpu_num
        total_throughputs.append(actual_total_throughput)
    
    # Plot Total Throughput
    plt.plot(gpu_nums, total_throughputs, marker='s', color='#4169E1', linewidth=3, markersize=8, label='Actual')
    ideal_throughputs = [single_gpu_total * gpu_num for gpu_num in gpu_nums]
    plt.plot(gpu_nums, ideal_throughputs, '--', color='#FF6B6B', linewidth=3, alpha=0.7, label='Ideal Linear')
    plt.xlabel("GPU Count", fontproperties=founders_reg_prop, fontsize=32)
    plt.ylabel("Total Throughput (GB/s)", fontproperties=founders_reg_prop, fontsize=32)
    plt.title("Total System Throughput", fontproperties=founders_reg_prop, fontsize=28)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    plt.xticks([1, 2, 3, 4], fontproperties=founders_reg_prop, fontsize=33)
    plt.yticks(fontproperties=founders_reg_prop, fontsize=32)
    
    # Create legend with larger font
    legend = plt.legend(prop=founders_reg_prop, fontsize=19)
    for text in legend.get_texts():
        text.set_fontproperties(founders_reg_prop)
        text.set_fontsize(19)
    plt.tight_layout()
    
    # check if the directory exists
    dir_path = os.path.dirname(save_filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(f"{save_filename}.pdf")
    plt.close()

def draw_relative_performance(gpu_nums, k_throughputs, v_throughputs, save_filename):
    """
    Draw relative performance compared to single GPU
    """
    plt.figure(figsize=(10, 5))
    
    # Calculate relative performance (normalized to single GPU = 1.0)
    k_relative = [k / k_throughputs[0] for k in k_throughputs]
    v_relative = [v / v_throughputs[0] for v in v_throughputs]
    total_relative = [(k_throughputs[i] + v_throughputs[i]) / (k_throughputs[0] + v_throughputs[0]) for i in range(len(gpu_nums))]
    
    # Plot relative performance
    plt.plot(gpu_nums, k_relative, marker='o', color='#1E90FF', linewidth=2, label='K Relative Performance')
    plt.plot(gpu_nums, v_relative, marker='s', color='#DC143C', linewidth=2, label='V Relative Performance') 
    plt.plot(gpu_nums, total_relative, marker='^', color='#228B22', linewidth=2, label='Total Relative Performance')
    
    # Add annotations showing percentage changes
    for i, gpu_num in enumerate(gpu_nums):
        if i > 0:  # Skip first point (baseline)
            k_change = (k_relative[i] - 1) * 100
            v_change = (v_relative[i] - 1) * 100
            total_change = (total_relative[i] - 1) * 100
            plt.annotate(f'{k_change:+.1f}%', (gpu_num, k_relative[i]), textcoords="offset points", 
                        xytext=(0,15), ha='center', fontproperties=founders_reg_prop, fontsize=16, color='#1E90FF')
            plt.annotate(f'{v_change:+.1f}%', (gpu_num, v_relative[i]), textcoords="offset points", 
                        xytext=(0,15), ha='center', fontproperties=founders_reg_prop, fontsize=16, color='#DC143C')
    
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Single GPU Baseline')
    plt.xlabel("GPU Count", fontproperties=founders_reg_prop, fontsize=32)
    plt.ylabel("Relative Performance", fontproperties=founders_reg_prop, fontsize=32)
    plt.title("Per-GPU Performance Relative to Single GPU", fontproperties=founders_reg_prop, fontsize=28)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    plt.xticks([1, 2, 3, 4], fontproperties=founders_reg_prop, fontsize=33)
    plt.yticks(fontproperties=founders_reg_prop, fontsize=32)
    
    # Create legend with larger font
    legend = plt.legend(prop=founders_reg_prop, fontsize=19)
    for text in legend.get_texts():
        text.set_fontproperties(founders_reg_prop)
        text.set_fontsize(19)
    plt.tight_layout()
    
    dir_path = os.path.dirname(save_filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(f"{save_filename}.pdf")
    plt.close()

def draw_throughput_multi_gpu_combined(gpu_nums, k_throughputs, v_throughputs, title, save_filename):
    """
    Draw combined K and V throughput vs GPU number plot as bar chart
    """
    plt.figure(figsize=(8, 5))  # Match the size of other plots
    
    # Color scheme: Blue for K, Red for V (similar to throughput.py)
    k_color = '#1E90FF'  # Dark blue for K
    v_color = '#DC143C'  # Dark red for V
    
    # Set up bar positions
    x = range(len(gpu_nums))
    width = 0.35  # Width of bars
    
    # Create bar chart
    bars1 = plt.bar([i - width/2 for i in x], k_throughputs, width, 
                    color=k_color, label='K Throughput', alpha=0.8)
    bars2 = plt.bar([i + width/2 for i in x], v_throughputs, width,
                    color=v_color, label='V Throughput', alpha=0.8)
    
    # Add value labels on top of bars
    for bar, value in zip(bars1, k_throughputs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{value:.1f}', ha='center', va='bottom', 
                fontproperties=founders_reg_prop, fontsize=19)
    
    for bar, value in zip(bars2, v_throughputs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{value:.1f}', ha='center', va='bottom',
                fontproperties=founders_reg_prop, fontsize=19)
    
    # Apply the specified font and increase font size for labels
    plt.xlabel("GPU Count", fontproperties=founders_reg_prop, fontsize=32)
    plt.ylabel("Throughput per GPU (GB/s)", fontproperties=founders_reg_prop, fontsize=32)
    plt.title(title, fontproperties=founders_reg_prop, fontsize=28)
    
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.7, axis='y')
    # Set x-axis to show only integer values from 1 to 4
    plt.xticks(x, gpu_nums, fontproperties=founders_reg_prop, fontsize=33)
    plt.yticks(fontproperties=founders_reg_prop, fontsize=32)
    # Set y-axis to start from 0 with some padding at the top
    max_throughput = max(max(k_throughputs), max(v_throughputs))
    plt.ylim(0, max_throughput * 1.15)  # Add 15% padding above the maximum value for labels
    
    # Create legend with larger font
    legend = plt.legend(prop=founders_reg_prop, fontsize=19)
    for text in legend.get_texts():
        text.set_fontproperties(founders_reg_prop)
        text.set_fontsize(19)
    plt.tight_layout()  # Adjust layout to prevent labels overlapping
    # check if the directory exists, get dir first
    dir_path = os.path.dirname(save_filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(f"{save_filename}.pdf")
    plt.close()

# Extract data for plotting
gpu_nums = [data['gpu_num'] for data in gpu_num_throughput_map.values()]
k_throughputs = [data['k_throughput'] for data in gpu_num_throughput_map.values()]
v_throughputs = [data['v_throughput'] for data in gpu_num_throughput_map.values()]

# Create all visualization plots
print("Generating multiple visualization plots...")

# 1. Combined K and V throughput bar chart
draw_throughput_multi_gpu_combined(
    gpu_nums, 
    k_throughputs,
    v_throughputs,
    "K and V Throughput vs GPU Number", 
    "./throughput_multi_gpu/a100_kv"
)

# 2. Scaling efficiency plot
draw_scaling_efficiency(
    gpu_nums,
    k_throughputs, 
    v_throughputs,
    "./throughput_multi_gpu/a100_scaling_efficiency"
)

# 3. Total throughput plot
draw_total_throughput(
    gpu_nums,
    k_throughputs,
    v_throughputs,
    "./throughput_multi_gpu/a100_total_throughput"
)

# 4. Relative performance plot
draw_relative_performance(
    gpu_nums,
    k_throughputs,
    v_throughputs, 
    "./throughput_multi_gpu/a100_relative_performance"
)

print("Generated multiple visualization PDFs:")
print("- throughput_multi_gpu_a100_kv.pdf (K and V throughput bar chart)")
print("- throughput_multi_gpu_a100_scaling_efficiency.pdf (Scaling efficiency)")
print("- throughput_multi_gpu_a100_total_throughput.pdf (Total system throughput)")
print("- throughput_multi_gpu_a100_relative_performance.pdf (Relative performance vs single GPU)")
