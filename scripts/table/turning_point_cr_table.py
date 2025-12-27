#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.serialization import load
from utils.compute import QuantMode

save_path = "../data/turning_point/turning_points.pkl"
turning_points = load(save_path)

print(turning_points[0])

model_name_map = {
    "meta-llama/Llama-2-7b-hf": "Llama-2-7B",
    "meta-llama/Llama-3.1-8B": "Llama-3.1-8B", 
    "meta-llama/Llama-2-13b-hf": "Llama-2-13B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "R1-Llama-8B",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral-8B",
    "microsoft/phi-4": "Phi-4"
}

benchmark_name_map = {
    "coqa": "CoQA",
    "gsm8k": "GSM8K",
    "mmlu": "MMLU",
    "gpqa": "GPQA_D",
    "squad_completion": "SQuAD_C",
    "winogrande": "Winogrande"
}

def escape_latex(text):
    """Escape special characters for LaTeX"""
    return text.replace('_', '\\_')

# Organize data by K/V, benchmark, model and quantization mode
data_by_kv_benchmark = {}
for item in turning_points:
    kv_type = "K" if item['is_k'] else "V"
    benchmark = item['benchmark']
    model = item['model']
    quant_mode = item['quantization_mode']
    
    if kv_type not in data_by_kv_benchmark:
        data_by_kv_benchmark[kv_type] = {}
    if benchmark not in data_by_kv_benchmark[kv_type]:
        data_by_kv_benchmark[kv_type][benchmark] = {}
    if model not in data_by_kv_benchmark[kv_type][benchmark]:
        data_by_kv_benchmark[kv_type][benchmark][model] = {}
    
    data_by_kv_benchmark[kv_type][benchmark][model][quant_mode] = item

# Get unique benchmarks and models
all_benchmarks = set()
all_models = set()
for kv_data in data_by_kv_benchmark.values():
    for benchmark, model_data in kv_data.items():
        all_benchmarks.add(benchmark)
        all_models.update(model_data.keys())

benchmarks = sorted(all_benchmarks)
models = sorted(all_models)

# Generate LaTeX tables - separate tables for K and V
for kv_type in ["K", "V"]:
    if kv_type in data_by_kv_benchmark:
        # Collect improvement data for averaging
        improvements_by_benchmark = {}  # {benchmark: [improvements]}
        improvements_by_model = {model: [] for model in models}  # {model: [improvements]}
        all_improvements = []
        
        print("\\begin{table*}[h]")
        print("\\centering")
        print("\\footnotesize")  # Make font smaller
        if kv_type == "K":
            print("\\begin{tabular}{|l|l|l|" + "c|" * len(models) + "c|}")
        else:  # V table has fewer columns
            print("\\begin{tabular}{|l|" + "c|" * len(models) + "c|}")
        print("\\hline")

        # Header row
        if kv_type == "K":
            header = "Benchmark & Quant & Range"
        else:  # V table doesn't need Quant and Range columns
            header = "Benchmark"
        for model in models:
            model_name = model_name_map.get(model, model)
            header += f" & {model_name}"
        header += " & Avg \\\\"
        print(header)
        print("\\hline")

        # Data rows
        kv_benchmarks = sorted(data_by_kv_benchmark[kv_type].keys())
        
        # Define quantization modes for each K/V type
        if kv_type == "K":
            quant_modes = [QuantMode.ChannelQuant, QuantMode.TokenQuant]
        else:  # V
            quant_modes = [QuantMode.TokenQuant]
        
        for i, benchmark in enumerate(kv_benchmarks):
            benchmark_display = benchmark_name_map.get(benchmark, benchmark)
            escaped_benchmark = escape_latex(benchmark_display)
            
            for j, quant_mode in enumerate(quant_modes):
                # Get range for this benchmark and quant mode (should be same for all models)
                benchmark_range = None
                ranges_for_benchmark = []
                
                for model in models:
                    if (model in data_by_kv_benchmark[kv_type][benchmark] and 
                        quant_mode in data_by_kv_benchmark[kv_type][benchmark][model]):
                        item = data_by_kv_benchmark[kv_type][benchmark][model][quant_mode]
                        range_data = item['relative_quantization_scale_range']
                        ranges_for_benchmark.append(range_data)
                        if benchmark_range is None:
                            benchmark_range = range_data
                
                # Check if all ranges are the same
                if ranges_for_benchmark:
                    all_same = all(r == benchmark_range for r in ranges_for_benchmark)
                    if not all_same:
                        print(f"Warning: Range not consistent for {kv_type}-{benchmark}-{quant_mode}: {ranges_for_benchmark}")
                    
                    range_str = f"[{benchmark_range[0]:.2f}, {benchmark_range[1]:.2f}]"
                else:
                    range_str = ""
                
                # Create row labels
                quant_mode_str = "Channel" if quant_mode == QuantMode.ChannelQuant else "Token"
                
                if kv_type == "K":
                    if j == 0:
                        benchmark_label = escaped_benchmark
                    else:
                        benchmark_label = ""  # Empty for second row
                else:  # V only has Token
                    benchmark_label = escaped_benchmark
                
                # Handle empty range string
                if not range_str:
                    range_str = ""
                
                if kv_type == "K":
                    row = f"{benchmark_label} & {quant_mode_str} & {range_str}"
                else:  # V table doesn't need Quant and Range columns
                    row = f"{benchmark_label}"
                
                # Collect improvements for this row
                row_improvements = []
                
                for model in models:
                    if (model in data_by_kv_benchmark[kv_type][benchmark] and 
                        quant_mode in data_by_kv_benchmark[kv_type][benchmark][model]):
                        item = data_by_kv_benchmark[kv_type][benchmark][model][quant_mode]
                        optimal_cr = item['optimal_cr']
                        
                        if kv_type == "K":
                            # For K table, show optimal_cr for Channel, and optimal_cr with improvement for Token
                            if quant_mode == QuantMode.ChannelQuant:
                                row += f" & {optimal_cr:.2f}"
                            else:  # TokenQuant
                                # Get corresponding Channel value for improvement calculation
                                if (model in data_by_kv_benchmark[kv_type][benchmark] and 
                                    QuantMode.ChannelQuant in data_by_kv_benchmark[kv_type][benchmark][model]):
                                    channel_item = data_by_kv_benchmark[kv_type][benchmark][model][QuantMode.ChannelQuant]
                                    channel_cr = channel_item['optimal_cr']
                                    improvement = (optimal_cr / channel_cr - 1) * 100  # token/channel - 1
                                    row_improvements.append(improvement)
                                    improvements_by_model[model].append(improvement)
                                    all_improvements.append(improvement)
                                    if improvement >= 0:
                                        row += f" & {optimal_cr:.2f}(+{improvement:.1f}\\%)"
                                    else:
                                        row += f" & {optimal_cr:.2f}({improvement:.1f}\\%)"
                                else:
                                    # If no channel data available, just show the value
                                    row += f" & {optimal_cr:.2f}"
                        else:
                            # For V table, show quant_cr/optimal_cr(improvement)
                            quant_cr = item['quant_cr']
                            improvement = (optimal_cr / quant_cr - 1) * 100  # percentage improvement (encode/quant - 1)
                            row_improvements.append(improvement)
                            improvements_by_model[model].append(improvement)
                            all_improvements.append(improvement)
                            if improvement >= 0:
                                row += f" & {quant_cr:.2f}/{optimal_cr:.2f}(+{improvement:.1f}\\%)"
                            else:
                                row += f" & {quant_cr:.2f}/{optimal_cr:.2f}({improvement:.1f}\\%)"
                    else:
                        row += " & "  # Leave blank if no data
                
                # Add row average for improvements (only for rows that have improvements)
                if row_improvements:
                    row_avg = sum(row_improvements) / len(row_improvements)
                    if kv_type == "K" and quant_mode == QuantMode.TokenQuant:
                        # Store benchmark average for K TokenQuant
                        improvements_by_benchmark[benchmark] = row_improvements
                    elif kv_type == "V":
                        # Store benchmark average for V
                        improvements_by_benchmark[benchmark] = row_improvements
                    
                    if row_avg >= 0:
                        row += f" & +{row_avg:.1f}\\%"
                    else:
                        row += f" & {row_avg:.1f}\\%"
                else:
                    row += " & "  # Leave blank if no improvements
                
                row += " \\\\"
                print(row)
            
            if i < len(kv_benchmarks) - 1:
                print("\\hline")

        # Add model averages row
        print("\\hline")
        if kv_type == "K":
            avg_row = "Avg & & "
        else:
            avg_row = "Avg"
        
        model_averages = []
        for model in models:
            if improvements_by_model[model]:
                model_avg = sum(improvements_by_model[model]) / len(improvements_by_model[model])
                model_averages.append(model_avg)
                if model_avg >= 0:
                    avg_row += f" & +{model_avg:.1f}\\%"
                else:
                    avg_row += f" & {model_avg:.1f}\\%"
            else:
                avg_row += " & "
        
        # Add overall average
        if all_improvements:
            overall_avg = sum(all_improvements) / len(all_improvements)
            if overall_avg >= 0:
                avg_row += f" & \\cdg{{+{overall_avg:.1f}\\%}}"
            else:
                avg_row += f" & \\cdg{{{overall_avg:.1f}\\%}}"
        else:
            avg_row += " & "
        
        avg_row += " \\\\"
        print(avg_row)
        
        print("\\hline")
        print("\\end{tabular}")
        
        # Different caption and label for K and V
        if kv_type == "K":
            caption = "Key Optimal Compression Ratios by Benchmark, Quantization Mode and Model"
        else:
            caption = "Value Optimal Compression Ratios by Benchmark and Model (Token Quantization)"
        print(f"\\caption{{{caption}}}")
        print(f"\\label{{tab:optimal_cr_{kv_type.lower()}}}")
        print("\\end{table*}")
        print()  # Add blank line between tables
        
        # Print additional statistics to terminal
        print(f"=== {kv_type} Table Statistics ===")
        
        if kv_type == "K":
            # Collect K statistics
            channel_crs = []
            token_crs = []
            k_improvements = []
            
            for benchmark in kv_benchmarks:
                for model in models:
                    # Channel CR
                    if (model in data_by_kv_benchmark[kv_type][benchmark] and 
                        QuantMode.ChannelQuant in data_by_kv_benchmark[kv_type][benchmark][model]):
                        channel_item = data_by_kv_benchmark[kv_type][benchmark][model][QuantMode.ChannelQuant]
                        channel_crs.append(channel_item['optimal_cr'])
                    
                    # Token CR and improvement
                    if (model in data_by_kv_benchmark[kv_type][benchmark] and 
                        QuantMode.TokenQuant in data_by_kv_benchmark[kv_type][benchmark][model]):
                        token_item = data_by_kv_benchmark[kv_type][benchmark][model][QuantMode.TokenQuant]
                        token_crs.append(token_item['optimal_cr'])
                        
                        # Calculate improvement
                        if (model in data_by_kv_benchmark[kv_type][benchmark] and 
                            QuantMode.ChannelQuant in data_by_kv_benchmark[kv_type][benchmark][model]):
                            channel_item = data_by_kv_benchmark[kv_type][benchmark][model][QuantMode.ChannelQuant]
                            improvement = (token_item['optimal_cr'] / channel_item['optimal_cr'] - 1) * 100
                            k_improvements.append(improvement)
            
            # Print K statistics
            if channel_crs:
                print(f"Channel Avg CR: {sum(channel_crs) / len(channel_crs):.2f}")
            if token_crs:
                print(f"Token Avg CR: {sum(token_crs) / len(token_crs):.2f}")
            if k_improvements:
                print(f"Improvement Avg: {sum(k_improvements) / len(k_improvements):.1f}%")
        
        else:  # V table
            # Collect V statistics
            quant_crs = []
            encode_crs = []
            v_improvements = []
            
            for benchmark in kv_benchmarks:
                for model in models:
                    if (model in data_by_kv_benchmark[kv_type][benchmark] and 
                        QuantMode.TokenQuant in data_by_kv_benchmark[kv_type][benchmark][model]):
                        item = data_by_kv_benchmark[kv_type][benchmark][model][QuantMode.TokenQuant]
                        quant_crs.append(item['quant_cr'])
                        encode_crs.append(item['optimal_cr'])
                        
                        # Calculate improvement
                        improvement = (item['optimal_cr'] / item['quant_cr'] - 1) * 100
                        v_improvements.append(improvement)
            
            # Print V statistics
            if quant_crs:
                print(f"Quant Avg CR: {sum(quant_crs) / len(quant_crs):.2f}")
            if encode_crs:
                print(f"Encode Avg CR: {sum(encode_crs) / len(encode_crs):.2f}")
            if v_improvements:
                print(f"Improvement Avg: {sum(v_improvements) / len(v_improvements):.1f}%")
        
        print()  # Add blank line
