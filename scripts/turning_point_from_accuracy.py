#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os
import numpy as np
# Remove matplotlib imports since we're not plotting anymore
# import matplotlib.pyplot as plt
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import PackKVCacheConfig
from utils.serialization import load, save
from utils.util import get_logger, block_other_logger, register_notify
from utils.compute import QuantMode
# Remove matplotlib font imports
# from matplotlib.font_manager import FontProperties

# font_path = 'Founders_Grotesk/FoundersGrotesk-Regular.otf'
# founders_reg_prop = FontProperties(fname=font_path)

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

def fit_curve(x, y, kv_type, quant_type, model_name, benchmark):
    """
    Perform cubic spline curve fitting and return the results
    Returns: dictionary with fitting results and metadata
    """
    x = np.array(x)
    y = np.array(y)
    
    x_indices = range(len(x))
    
    # Add cubic spline interpolation
    fitted_x = None
    fitted_y = None
    spline_func = None
    
    if len(x) >= 3:  # Need at least 3 points for fitting
        from scipy.interpolate import CubicSpline
        
        # Use CubicSpline which passes through all data points while maintaining smoothness
        spline_func = CubicSpline(x_indices, y, bc_type='natural')
        
        # Generate smooth curve points
        fitted_x = np.linspace(0, len(x) - 1, 200)
        fitted_y = spline_func(fitted_x)
    
    # Calculate baseline value (95% of first y value)
    baseline_value = 0.95 * y[0]
    
    return {
        'kv_type': kv_type,
        'quant_type': quant_type,
        'model_name': model_name,
        'benchmark': benchmark,
        'original_x': x,
        'original_y': y,
        'fitted_x': fitted_x,
        'fitted_y': fitted_y,
        'spline_function': spline_func,
        'baseline_value': baseline_value
    }

logger = get_logger(__file__)
block_other_logger(logger)
setting_path = "data/accuracy/accuracy_setting_map.pkl"
setting_map: Dict[int, Tuple[str, PackKVCacheConfig]] = load(setting_path)

save_path = "data/accuracy/accuracy_result_map.pkl"
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

# Store all fitting results
all_fitting_results = []

for model in model_list:
    for benchmark in benchmark_list:
        model_name_only = model.split('/')[-1] if '/' in model else model

        # K channel quant
        filtered_pairs = filter(pairs, lambda setting, _: setting[1].model_name == model and setting[0] == benchmark and setting[1].quant_method.value[0] == QuantMode.ChannelQuant)
        scales, accuracies = extract_scales_accuracies(filtered_pairs, benchmark, True)
        if scales and accuracies:  # Only process if we have data
            result = fit_curve(scales, accuracies, "K", "Channel Quant", model_name_only, "gpqa" if isinstance(benchmark, list) else benchmark)
            result['model_full_name'] = model  # Store full model name
            result['quantization_mode'] = QuantMode.ChannelQuant
            all_fitting_results.append(result)
        
        # K token quant
        filtered_pairs = filter(pairs, lambda setting, _: setting[1].model_name == model and setting[0] == benchmark and setting[1].quant_method.value[0] == QuantMode.TokenQuant and setting[1].v_quant_scale_rel == 0.01)
        scales, accuracies = extract_scales_accuracies(filtered_pairs, benchmark, True)
        if scales and accuracies:  # Only process if we have data
            result = fit_curve(scales, accuracies, "K", "Token Quant", model_name_only, "gpqa" if isinstance(benchmark, list) else benchmark)
            result['model_full_name'] = model  # Store full model name
            result['quantization_mode'] = QuantMode.TokenQuant
            all_fitting_results.append(result)
        
        # V token quant
        filtered_pairs = filter(pairs, lambda setting, _: setting[1].model_name == model and setting[0] == benchmark and setting[1].quant_method.value[1] == QuantMode.TokenQuant and setting[1].quant_method.value[0] == QuantMode.TokenQuant and setting[1].k_quant_scale_rel == 0.01)
        scales, accuracies = extract_scales_accuracies(filtered_pairs, benchmark, False)
        if scales and accuracies:  # Only process if we have data
            result = fit_curve(scales, accuracies, "V", "Token Quant", model_name_only, "gpqa" if isinstance(benchmark, list) else benchmark)
            result['model_full_name'] = model  # Store full model name
            result['quantization_mode'] = QuantMode.TokenQuant
            all_fitting_results.append(result)

# Output all fitting results
print("Fitting Results:")
print("=" * 80)

save_path = "data/turning_point/turning_points.pkl"
turning_points = []

for i, result in enumerate(all_fitting_results):
    print(f"KV Type: {result['kv_type']}")
    print(f"Quantization Type: {result['quant_type']}")
    print(f"Model Name: {result['model_name']}")
    print(f"Benchmark: {result['benchmark']}")
    print(f"Baseline Value (95%): {result['baseline_value']:.6f}")
    
    print(f"\nOriginal Data:")
    print(f"X (Quantization Scale): {result['original_x']}")
    print(f"Y (Accuracy): {result['original_y']}")
    
    # Initialize turning point dictionary
    turning_point_data = {
        "is_k": result['kv_type'] == "K",
        "model": result['model_full_name'],
        "benchmark": result['benchmark'],
        "quantization_mode": result['quantization_mode'],
        "95%accuracy": float(result['baseline_value']),  # Convert numpy float64 to float
        "relative_quantization_scale_range": [float(result['original_x'][0]), float(result['original_x'][-1])],
        "sensitive": None,
        "turning_point": None
    }
    
    if result['fitted_x'] is not None and result['fitted_y'] is not None and result['spline_function'] is not None:
        print(f"\nUsing Cubic Spline Interpolation")
        
        # Solve for x when y = 0.95 * y[0] (baseline value)
        target_y = result['baseline_value']
        spline_func = result['spline_function']
        original_x_array = result['original_x']
        original_y_array = result['original_y']
        
        # Check if the last y value (at maximum quantization) is above 95% baseline (quantization insensitive)
        last_y_value = original_y_array[-1]
        
        if last_y_value >= target_y:
            # Quantization insensitive: turning point is the last x value
            turning_point_x = original_x_array[-1]
            turning_point_data["sensitive"] = False
            turning_point_data["turning_point"] = float(turning_point_x)
            print(f"Quantization insensitive (accuracy at max quantization {last_y_value:.6f} >= 95% baseline {target_y:.6f})")
            print(f"Turning Point: x = {turning_point_x:.6f} (last x value)")
        else:
            # Quantization sensitive: find all roots and choose the maximum one
            turning_point_data["sensitive"] = True
            print(f"Quantization sensitive (accuracy at max quantization {last_y_value:.6f} < 95% baseline {target_y:.6f})")
            
            # Define the function to find roots of: spline(x) - target_y = 0
            def target_function(x_idx):
                return spline_func(x_idx) - target_y
            
            # Search for all turning points using numerical methods
            x_min = 0  # Since we use x_indices starting from 0
            x_max = len(original_x_array) - 1
            
            turning_point_indices = []
            
            # Try to find all crossings by checking multiple intervals
            try:
                from scipy.optimize import brentq
                
                # Sample many points to detect sign changes
                test_indices = np.linspace(x_min, x_max, 1000)
                test_values = [target_function(x) for x in test_indices]
                
                # Find all intervals where sign changes occur
                for i in range(len(test_values) - 1):
                    if test_values[i] * test_values[i + 1] < 0:  # Sign change detected
                        try:
                            root = brentq(target_function, test_indices[i], test_indices[i + 1])
                            turning_point_indices.append(root)
                        except:
                            continue
                
                # Remove duplicate roots (within small tolerance)
                if turning_point_indices:
                    turning_point_indices = sorted(set([round(x, 6) for x in turning_point_indices]))
                    
            except Exception as e:
                print(f"Error in numerical root finding: {e}")
                turning_point_indices = []
            
            if turning_point_indices:
                # Choose the maximum root
                turning_point_index = max(turning_point_indices)
                print(f"Found {len(turning_point_indices)} crossing points, choosing maximum: {turning_point_index:.6f}")
                
                # Convert index back to original x scale
                if turning_point_index < len(original_x_array):
                    # Linear interpolation for more precise x value
                    if turning_point_index == int(turning_point_index):
                        turning_point_x = original_x_array[int(turning_point_index)]
                    else:
                        idx_low = int(np.floor(turning_point_index))
                        idx_high = int(np.ceil(turning_point_index))
                        if idx_high < len(original_x_array):
                            weight = turning_point_index - idx_low
                            turning_point_x = (1 - weight) * original_x_array[idx_low] + weight * original_x_array[idx_high]
                        else:
                            turning_point_x = original_x_array[idx_low]
                    
                    turning_point_data["turning_point"] = float(turning_point_x)
                    print(f"Turning Point (when accuracy = 95% baseline): x = {turning_point_x:.6f}")
                else:
                    print("Turning point index out of range")
            else:
                print("No crossing points found - benchmark not sensitive to quantization")
    else:
        print(f"\nInsufficient data points for fitting (need at least 3 points)")
        turning_point_data["sensitive"] = None
        turning_point_data["turning_point"] = None
    
    # Add the turning point data to the list
    turning_points.append(turning_point_data)
    
    print("-" * 60)

print(f"\nTotal processed {len(all_fitting_results)} data groups")

# Save turning points data
save(turning_points, save_path)
print(f"\nTurning points data saved to: {save_path}")
print(f"Saved {len(turning_points)} turning points")

# Print summary of turning points
print("\nTurning Points Summary:")
print("=" * 80)
for i, tp in enumerate(turning_points):
    print(f"Entry {i+1}:")
    print(f"  Model: {tp['model']}")
    print(f"  Benchmark: {tp['benchmark']}")
    print(f"  KV Type: {'K' if tp['is_k'] else 'V'}")
    print(f"  Quantization Mode: {tp['quantization_mode']}")
    print(f"  95% Accuracy: {tp['95%accuracy']:.6f}")
    print(f"  Quantization Scale Range: {tp['relative_quantization_scale_range']}")
    print(f"  Sensitive: {tp['sensitive']}")
    print(f"  Turning Point: {tp['turning_point']}")
    print()