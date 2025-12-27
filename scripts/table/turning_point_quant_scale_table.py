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
        print("\\begin{table*}[h]")
        print("\\centering")
        print("\\begin{tabular}{|l|l|l|" + "c|" * len(models) + "}")
        print("\\hline")

        # Header row
        header = "Benchmark & Quant Mode & Range"
        for model in models:
            model_name = model_name_map.get(model, model)
            header += f" & {model_name}"
        header += " \\\\"
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
                
                row = f"{benchmark_label} & {quant_mode_str} & {range_str}"
                
                for model in models:
                    if (model in data_by_kv_benchmark[kv_type][benchmark] and 
                        quant_mode in data_by_kv_benchmark[kv_type][benchmark][model]):
                        item = data_by_kv_benchmark[kv_type][benchmark][model][quant_mode]
                        turning_point = item['turning_point']
                        row += f" & {turning_point:.4f}"
                    else:
                        row += " & "  # Leave blank if no data
                row += " \\\\"
                print(row)
            
            if i < len(kv_benchmarks) - 1:
                print("\\hline")

        print("\\hline")
        print("\\end{tabular}")
        
        # Different caption and label for K and V
        if kv_type == "K":
            caption = "Key Turning Points by Benchmark, Quantization Mode and Model"
        else:
            caption = "Value Turning Points by Benchmark and Model (Token Quantization)"
        print(f"\\caption{{{caption}}}")
        print(f"\\label{{tab:turning_points_{kv_type.lower()}}}")
        print("\\end{table*}")
        print()  # Add blank line between tables
