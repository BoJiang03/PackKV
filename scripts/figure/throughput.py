#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import PackKVCacheConfig
from utils.serialization import load, save
from utils.util import get_logger, block_other_logger, register_notify
from utils.compute import QuantMode
import numpy as np
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
a100_setting_path = "../data/throughput/throughput_setting_map_a100.pkl"
a100_setting_map: Dict[int, PackKVCacheConfig] = load(a100_setting_path)
rtx_pro_setting_path = "../data/throughput/throughput_setting_map_rtx_pro.pkl"
rtx_pro_setting_map: Dict[int, PackKVCacheConfig] = load(rtx_pro_setting_path)

a100_result_path = "../data/throughput/throughput_result_map_a100.pkl"
a100_result_map = load(a100_result_path)
rtx_pro_result_path = "../data/throughput/throughput_result_map_rtx_pro.pkl"
rtx_pro_result_map = load(rtx_pro_result_path)

a100_pairs = pair(a100_setting_map.values(), a100_result_map.values())
rtx_pro_pairs = pair(rtx_pro_setting_map.values(), rtx_pro_result_map.values())

def filter(pairs, func):
    filtered_pairs = []
    for pair in pairs:
        setting, result = pair
        if func(setting, result):
            filtered_pairs.append(pair)
    return filtered_pairs

a100_llama_filtered_pairs = filter(a100_pairs, lambda setting, result: setting[1].model_name == "meta-llama/Llama-3.1-8B")
a100_ministral_filtered_pairs = filter(a100_pairs, lambda setting, result: setting[1].model_name == "mistralai/Ministral-8B-Instruct-2410")
rtx_pro_llama_filtered_pairs = filter(rtx_pro_pairs, lambda setting, result: setting[1].model_name == "meta-llama/Llama-3.1-8B")
rtx_pro_ministral_filtered_pairs = filter(rtx_pro_pairs, lambda setting, result: setting[1].model_name == "mistralai/Ministral-8B-Instruct-2410")

def process_pairs(pairs):
    processed_pairs = []
    for setting, result_list in pairs:
        # Extract ctx_len from setting (first element of the tuple)
        ctx_len, config = setting
        
        # result_list is a list containing one dictionary
        result = result_list[0]
        
        # Sum original sizes
        k_original_size_sum = sum(result['k_original_size'])
        v_original_size_sum = sum(result['v_original_size'])
        
        # Sum kernel times
        k_our_kernel_time_sum = sum(result['k_our_kernel_time'])
        k_our_none_kernel_time_sum = sum(result['k_our_none_kernel_time'])
        k_pytorch_kernel_time_sum = sum(result['k_pytorch_kernel_time'])
        v_our_kernel_time_sum = sum(result['v_our_kernel_time'])
        v_our_none_kernel_time_sum = sum(result['v_our_none_kernel_time'])
        v_pytorch_kernel_time_sum = sum(result['v_pytorch_kernel_time'])
        
        # Create new result with summed values and ctx_len
        new_result = {
            'ctx_len': ctx_len,
            'k_original_size_sum': k_original_size_sum,
            'v_original_size_sum': v_original_size_sum,
            'k_our_kernel_time_sum': k_our_kernel_time_sum,
            'k_our_none_kernel_time_sum': k_our_none_kernel_time_sum,
            'k_pytorch_kernel_time_sum': k_pytorch_kernel_time_sum,
            'v_our_kernel_time_sum': v_our_kernel_time_sum,
            'v_our_none_kernel_time_sum': v_our_none_kernel_time_sum,
            'v_pytorch_kernel_time_sum': v_pytorch_kernel_time_sum
        }
        
        processed_pairs.append((ctx_len, new_result))
    
    return processed_pairs

def pairs_to_arrays(pairs):
    """
    Convert pairs to arrays for plotting
    Returns: dict with arrays
    """
    ctx_lens = []
    k_data_sizes = []
    v_data_sizes = []
    k_our_kernel_times = []
    v_our_kernel_times = []
    k_our_none_kernel_times = []
    v_our_none_kernel_times = []
    k_pytorch_kernel_times = []
    v_pytorch_kernel_times = []
    
    for ctx_len, result in pairs:
        ctx_lens.append(ctx_len)
        k_data_sizes.append(result['k_original_size_sum'])
        v_data_sizes.append(result['v_original_size_sum'])
        k_our_kernel_times.append(result['k_our_kernel_time_sum'])
        v_our_kernel_times.append(result['v_our_kernel_time_sum'])
        k_our_none_kernel_times.append(result['k_our_none_kernel_time_sum'])
        v_our_none_kernel_times.append(result['v_our_none_kernel_time_sum'])
        k_pytorch_kernel_times.append(result['k_pytorch_kernel_time_sum'])
        v_pytorch_kernel_times.append(result['v_pytorch_kernel_time_sum'])
    
    # Calculate total kernel time (our kernel + our none kernel)
    k_our_total_kernel_times = [k_time + k_none_time for k_time, k_none_time in zip(k_our_kernel_times, k_our_none_kernel_times)]
    v_our_total_kernel_times = [v_time + v_none_time for v_time, v_none_time in zip(v_our_kernel_times, v_our_none_kernel_times)]
    
    # Calculate throughput (data size / kernel time)
    # Note: kernel time is in ms, data size is in bytes
    # Convert to GB/s: (bytes/ms) * (1000 ms/s) / (1024^3 bytes/GB)
    bytes_to_gb = 1024**3  # 1073741824
    ms_to_s = 1000
    conversion_factor = ms_to_s / bytes_to_gb
    
    k_our_throughput = [(size / time) * conversion_factor for size, time in zip(k_data_sizes, k_our_kernel_times)]
    v_our_throughput = [(size / time) * conversion_factor for size, time in zip(v_data_sizes, v_our_kernel_times)]
    k_our_none_throughput = [(size / time) * conversion_factor for size, time in zip(k_data_sizes, k_our_none_kernel_times)]
    v_our_none_throughput = [(size / time) * conversion_factor for size, time in zip(v_data_sizes, v_our_none_kernel_times)]
    k_our_total_throughput = [(size / time) * conversion_factor for size, time in zip(k_data_sizes, k_our_total_kernel_times)]
    v_our_total_throughput = [(size / time) * conversion_factor for size, time in zip(v_data_sizes, v_our_total_kernel_times)]
    k_pytorch_throughput = [(size / time) * conversion_factor for size, time in zip(k_data_sizes, k_pytorch_kernel_times)]
    v_pytorch_throughput = [(size / time) * conversion_factor for size, time in zip(v_data_sizes, v_pytorch_kernel_times)]
    
    return {
        'ctx_lens': ctx_lens,
        'k_data_sizes': k_data_sizes,
        'v_data_sizes': v_data_sizes,
        'k_our_kernel_times': k_our_kernel_times,
        'v_our_kernel_times': v_our_kernel_times,
        'k_our_none_kernel_times': k_our_none_kernel_times,
        'v_our_none_kernel_times': v_our_none_kernel_times,
        'k_our_total_kernel_times': k_our_total_kernel_times,
        'v_our_total_kernel_times': v_our_total_kernel_times,
        'k_pytorch_kernel_times': k_pytorch_kernel_times,
        'v_pytorch_kernel_times': v_pytorch_kernel_times,
        'k_our_throughput': k_our_throughput,
        'v_our_throughput': v_our_throughput,
        'k_our_none_throughput': k_our_none_throughput,
        'v_our_none_throughput': v_our_none_throughput,
        'k_our_total_throughput': k_our_total_throughput,
        'v_our_total_throughput': v_our_total_throughput,
        'k_pytorch_throughput': k_pytorch_throughput,
        'v_pytorch_throughput': v_pytorch_throughput
    }

a100_llama_filtered_pairs = process_pairs(a100_llama_filtered_pairs)
a100_ministral_filtered_pairs = process_pairs(a100_ministral_filtered_pairs)
rtx_pro_llama_filtered_pairs = process_pairs(rtx_pro_llama_filtered_pairs)
rtx_pro_ministral_filtered_pairs = process_pairs(rtx_pro_ministral_filtered_pairs)

# Convert to arrays for plotting
a100_llama_arrays = pairs_to_arrays(a100_llama_filtered_pairs)
a100_ministral_arrays = pairs_to_arrays(a100_ministral_filtered_pairs)
rtx_pro_llama_arrays = pairs_to_arrays(rtx_pro_llama_filtered_pairs)
rtx_pro_ministral_arrays = pairs_to_arrays(rtx_pro_ministral_filtered_pairs)

def draw_throughput_comparison(ctx_lens, k_pytorch_throughput, k_our_total_throughput, v_pytorch_throughput, v_our_total_throughput, title, save_filename):
    """
    Draw throughput comparison plot with both K and V data
    Use color families for K/V distinction and depth for method distinction
    """
    # Convert ctx_lens to "xx k" format
    ctx_lens_k = [ctx_len / 1024 for ctx_len in ctx_lens]
    
    plt.figure(figsize=(8, 5.5))

    # Color scheme: Blue family for K, Red family for V
    # Lighter shades for PyTorch, darker shades for Our method
    k_pytorch_color = '#87CEEB'  # Light blue for K PyTorch
    k_our_color = '#1E90FF'      # Dark blue for K Our
    v_pytorch_color = '#FFB6C1'  # Light red for V PyTorch
    v_our_color = '#DC143C'      # Dark red for V Our

    # Draw K throughput lines
    plt.plot(range(len(ctx_lens_k)), k_pytorch_throughput, marker='o', markersize=12, color=k_pytorch_color, linewidth=2, label='K cuBLAS Throughput')
    plt.plot(range(len(ctx_lens_k)), k_our_total_throughput, marker='o', markersize=12, color=k_our_color, linewidth=2, label='K PackKV Throughput')

    # Draw V throughput lines
    plt.plot(range(len(ctx_lens_k)), v_pytorch_throughput, marker='o', markersize=12, color=v_pytorch_color, linewidth=2, label='V cuBLAS Throughput')
    plt.plot(range(len(ctx_lens_k)), v_our_total_throughput, marker='o', markersize=12, color=v_our_color, linewidth=2, label='V PackKV Throughput')

    # Set x-axis labels with "xx k" format
    plt.xticks(range(len(ctx_lens_k)), [f'{ctx:.0f}k' for ctx in ctx_lens_k])

    # Apply the specified font and increase font size for labels
    plt.xlabel("Context Length", fontproperties=founders_reg_prop, fontsize=32)
    plt.ylabel("Throughput (GB/s)", fontproperties=founders_reg_prop, fontsize=32)
    plt.title(title, fontproperties=founders_reg_prop, fontsize=28)

    plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    # Increase font size for tick labels
    plt.xticks(fontproperties=founders_reg_prop, fontsize=33)
    plt.yticks(fontproperties=founders_reg_prop, fontsize=32)
    
    # Create legend with larger font
    legend = plt.legend(prop=founders_reg_prop, fontsize=19)
    for text in legend.get_texts():
        text.set_fontproperties(founders_reg_prop)
        text.set_fontsize(19)
    plt.tight_layout()  # Adjust layout to prevent labels overlapping

    plt.savefig(f"{save_filename}.pdf")
    plt.close()

# Create save directory if not exists
save_path = "./throughput"
os.makedirs(save_path, exist_ok=True)

# Generate all 4 plots (combined K and V)
datasets = [
    ("a100_llama", a100_llama_arrays, "A100", "Llama-3.1-8B"),
    ("a100_ministral", a100_ministral_arrays, "A100", "Ministral-8B"),
    ("rtx_pro_llama", rtx_pro_llama_arrays, "RTX Pro", "Llama-3.1-8B"),
    ("rtx_pro_ministral", rtx_pro_ministral_arrays, "RTX Pro", "Ministral-8B")
]

for dataset_name, arrays, gpu_name, model_name in datasets:
    # Combined K and V throughput plot
    draw_throughput_comparison(
        ctx_lens=arrays['ctx_lens'],
        k_pytorch_throughput=arrays['k_pytorch_throughput'],
        k_our_total_throughput=arrays['k_our_total_throughput'],
        v_pytorch_throughput=arrays['v_pytorch_throughput'],
        v_our_total_throughput=arrays['v_our_total_throughput'],
        title=f"{gpu_name} - {model_name}",
        save_filename=os.path.join(save_path, f"{dataset_name}_throughput")
    )

print("Generated 4 combined throughput comparison plots in ./throughput/")

def calculate_none_kernel_percentage(our_kernel_times, our_none_kernel_times):
    """
    Calculate the percentage of our none kernel time by summing all times first
    """
    total_our_kernel = sum(our_kernel_times)
    total_our_none_kernel = sum(our_none_kernel_times)
    total_time = total_our_kernel + total_our_none_kernel
    
    if total_time > 0:
        none_percentage = (total_our_none_kernel / total_time) * 100
        return none_percentage
    else:
        return 0.0

def create_kernel_time_pie_chart(our_kernel_times, our_none_kernel_times, title, filename):
    """
    Create pie chart showing the proportion of our kernel time vs none kernel time
    """
    # Sum all times across all context lengths
    total_our_kernel = sum(our_kernel_times)
    total_our_none_kernel = sum(our_none_kernel_times)
    
    labels = ['Our Kernel', 'Our None Kernel']
    sizes = [total_our_kernel, total_our_none_kernel]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0)  # explode the our kernel slice slightly
    
    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                                       shadow=True, startangle=90, textprops={'fontsize': 14})
    
    # Apply custom font to labels and percentages
    for text in texts:
        text.set_fontproperties(founders_reg_prop)
        text.set_fontsize(16)
    
    for autotext in autotexts:
        autotext.set_fontproperties(founders_reg_prop)
        autotext.set_fontsize(14)
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    plt.title(title, fontproperties=founders_reg_prop, fontsize=18, pad=20)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()

print("Generated 4 combined throughput comparison plots in ./throughput/")

# Generate pie charts for kernel time breakdown
# A100 pie charts
create_kernel_time_pie_chart(
    our_kernel_times=a100_llama_arrays['k_our_kernel_times'],
    our_none_kernel_times=a100_llama_arrays['k_our_none_kernel_times'],
    title="A100 - K Kernel Time Breakdown",
    filename="a100_k_kernel_time_pie.pdf"
)

create_kernel_time_pie_chart(
    our_kernel_times=a100_llama_arrays['v_our_kernel_times'],
    our_none_kernel_times=a100_llama_arrays['v_our_none_kernel_times'],
    title="A100 - V Kernel Time Breakdown",
    filename="a100_v_kernel_time_pie.pdf"
)

# RTX Pro pie charts
create_kernel_time_pie_chart(
    our_kernel_times=rtx_pro_llama_arrays['k_our_kernel_times'],
    our_none_kernel_times=rtx_pro_llama_arrays['k_our_none_kernel_times'],
    title="RTX Pro - K Kernel Time Breakdown",
    filename="rtx_pro_k_kernel_time_pie.pdf"
)

create_kernel_time_pie_chart(
    our_kernel_times=rtx_pro_llama_arrays['v_our_kernel_times'],
    our_none_kernel_times=rtx_pro_llama_arrays['v_our_none_kernel_times'],
    title="RTX Pro - V Kernel Time Breakdown",
    filename="rtx_pro_v_kernel_time_pie.pdf"
)

print("Generated 4 kernel time breakdown pie charts in ./throughput/")

# Calculate and print our none kernel percentages
print("\n" + "="*50)
print("OUR NONE KERNEL PERCENTAGE ANALYSIS")
print("="*50)

# A100 percentages
a100_k_none_percentage = calculate_none_kernel_percentage(
    a100_llama_arrays['k_our_kernel_times'],
    a100_llama_arrays['k_our_none_kernel_times']
)
a100_v_none_percentage = calculate_none_kernel_percentage(
    a100_llama_arrays['v_our_kernel_times'],
    a100_llama_arrays['v_our_none_kernel_times']
)

print(f"A100 K Our None Kernel Percentage: {a100_k_none_percentage:.1f}%")
print(f"A100 V Our None Kernel Percentage: {a100_v_none_percentage:.1f}%")

# RTX Pro percentages
rtx_k_none_percentage = calculate_none_kernel_percentage(
    rtx_pro_llama_arrays['k_our_kernel_times'],
    rtx_pro_llama_arrays['k_our_none_kernel_times']
)
rtx_v_none_percentage = calculate_none_kernel_percentage(
    rtx_pro_llama_arrays['v_our_kernel_times'],
    rtx_pro_llama_arrays['v_our_none_kernel_times']
)

print(f"RTX Pro K Our None Kernel Percentage: {rtx_k_none_percentage:.1f}%")
print(f"RTX Pro V Our None Kernel Percentage: {rtx_v_none_percentage:.1f}%")

# Overall averages
k_avg_none_percentage = (a100_k_none_percentage + rtx_k_none_percentage) / 2
v_avg_none_percentage = (a100_v_none_percentage + rtx_v_none_percentage) / 2
print(f"\nK Average Our None Kernel Percentage: {k_avg_none_percentage:.1f}%")
print(f"V Average Our None Kernel Percentage: {v_avg_none_percentage:.1f}%")

overall_avg_none_percentage = (k_avg_none_percentage + v_avg_none_percentage) / 2
print(f"Overall Average Our None Kernel Percentage: {overall_avg_none_percentage:.1f}%")

print("="*50)

# Calculate average performance improvement (percentage)
print("\n" + "="*50)
print("AVERAGE PERFORMANCE IMPROVEMENT ANALYSIS")
print("="*50)

def calculate_average_improvement(pytorch_times, total_times, name):
    """Calculate average performance improvement as percentage across all context lengths"""
    improvements = [(pytorch_time - total_time) / total_time * 100 for pytorch_time, total_time in zip(pytorch_times, total_times)]
    avg_improvement = sum(improvements) / len(improvements)
    print(f"{name}: {avg_improvement:.1f}%")
    return avg_improvement

# A100 improvements
print("\nA100 Improvements:")
a100_k_improvement = calculate_average_improvement(
    a100_llama_arrays['k_pytorch_kernel_times'], 
    a100_llama_arrays['k_our_total_kernel_times'], 
    "A100 K"
)
a100_v_improvement = calculate_average_improvement(
    a100_llama_arrays['v_pytorch_kernel_times'], 
    a100_llama_arrays['v_our_total_kernel_times'], 
    "A100 V"
)

# RTX Pro improvements
print("\nRTX Pro Improvements:")
rtx_k_improvement = calculate_average_improvement(
    rtx_pro_llama_arrays['k_pytorch_kernel_times'], 
    rtx_pro_llama_arrays['k_our_total_kernel_times'], 
    "RTX Pro K"
)
rtx_v_improvement = calculate_average_improvement(
    rtx_pro_llama_arrays['v_pytorch_kernel_times'], 
    rtx_pro_llama_arrays['v_our_total_kernel_times'], 
    "RTX Pro V"
)

# Overall averages
print("\nOverall Averages:")
k_avg_improvement = (a100_k_improvement + rtx_k_improvement) / 2
v_avg_improvement = (a100_v_improvement + rtx_v_improvement) / 2
print(f"K Average Improvement: {k_avg_improvement:.1f}%")
print(f"V Average Improvement: {v_avg_improvement:.1f}%")

overall_avg_improvement = (k_avg_improvement + v_avg_improvement) / 2
print(f"Overall Average Improvement: {overall_avg_improvement:.1f}%")

print("="*50)