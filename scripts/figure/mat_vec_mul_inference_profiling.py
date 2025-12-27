#! /usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
from tqdm import tqdm
# flash attention 2 for prefill, eager for decoding
from models.llama_for_profiling import LlamaForCausalLM
from evaluation.evaluation import get_ctx_len_text_from_wikitext_103_v1

# Add the parent directory to sys.path
from utils.util import get_logger, block_other_logger
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

logger = get_logger(__file__)
block_other_logger(logger)

# Font setup for plotting
def get_font_properties():
    """Get font properties with proper fallback for different systems"""
    # Try common font paths for different systems
    font_paths = [
        '/System/Library/Fonts/Helvetica.ttc',  # macOS
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',  # Linux
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
        '/Windows/Fonts/arial.ttf',  # Windows
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return fm.FontProperties(fname=font_path)
            except:
                continue
    
    # Fallback to system default fonts by name
    font_names = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    for font_name in font_names:
        try:
            return fm.FontProperties(family=font_name)
        except:
            continue
    
    # Final fallback to matplotlib default
    return fm.FontProperties()

founders_reg_prop = get_font_properties()

# Global set to track discovered kernel names across runs
discovered_kernels = set()

def create_matvec_pie_chart(matvec_time_ms, total_time_ms, ctx_len, save_dir):
    """
    Create a pie chart showing GPU kernel time proportion between 
    matrix-vector multiplication and other operations.
    """
    other_time_ms = total_time_ms - matvec_time_ms
    matvec_percentage = (matvec_time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
    other_percentage = 100 - matvec_percentage
    
    labels = ['Matrix-Vector Mul', 'Other Kernels']
    sizes = [matvec_percentage, other_percentage]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0)  # explode the matrix-vector slice slightly
    
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
    
    title = f'GPU Kernel Time Distribution\n(Context Length: {ctx_len//1024}K tokens)'
    plt.title(title, fontproperties=founders_reg_prop, fontsize=18, pad=20)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Add timing information as text below the chart
    info_text = f'Total GPU Time: {total_time_ms:.1f}ms\nMatrix-Vector Time: {matvec_time_ms:.1f}ms\nOther Kernels Time: {other_time_ms:.1f}ms'
    plt.figtext(0.5, 0.02, info_text, ha='center', fontproperties=founders_reg_prop, fontsize=12)
    
    plt.tight_layout()
    
    filename = f'matvec_gpu_time_ctx_{ctx_len//1024}k.pdf'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Pie chart saved to: {filepath}")
    return filepath

def analyze_matvec_kernels(prof, ctx_len):
    """
    Analyze profiling results to identify matrix-vector multiplication kernels
    and calculate their proportion of total GPU time.
    """
    global discovered_kernels
    
    logger.info("="*60)
    logger.info("ANALYZING MATRIX-VECTOR MULTIPLICATION KERNELS")
    logger.info("="*60)
    
    # Get profiling events
    events = prof.profiler.function_events
    
    # Patterns to identify matrix-vector multiplication kernels
    matvec_patterns = [
        r'.*gemv.*',           # General matrix-vector multiplication
        r'.*sgemv.*',          # Single precision GEMV
        r'.*dgemv.*',          # Double precision GEMV
        r'.*hgemv.*',          # Half precision GEMV
        r'.*bgemv.*',          # BFloat16 GEMV
        r'.*cutlass.*gemv.*',  # CUTLASS GEMV kernels
        r'.*cublas.*gemv.*',   # cuBLAS GEMV kernels
        r'.*cublasLt.*Gemv.*', # cuBLAS LT GEMV
        r'.*flash_attn.*gemv.*', # FlashAttention GEMV
        r'.*attention.*gemv.*',  # Attention GEMV
        r'.*linear.*',         # Linear layer operations (often GEMV in decode phase)
        r'.*addmm.*',          # Add + matrix multiplication
        r'.*mm.*',             # Matrix multiplication (could be GEMV when one dim is 1)
        r'.*bmm.*',            # Batch matrix multiplication
        r'.*matmul.*',         # General matrix multiplication
        r'.*dot.*',            # Dot product operations
        r'.*gemmk1.*',         # GEMM K1 kernels
    ]
    
    # Compile regex patterns for efficiency
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in matvec_patterns]
    
    # Collect GPU events and categorize them
    gpu_events = []
    matvec_events = []
    matvec_kernels = defaultdict(list)
    new_kernels = []
    
    total_gpu_time = 0
    total_matvec_time = 0
    
    for event in events:
        if event.device_type == torch.profiler.DeviceType.CUDA:
            # Fix deprecation warning: use device_time_total instead of cuda_time_total
            gpu_time = event.device_time_total if hasattr(event, 'device_time_total') else event.self_device_time_total
            if gpu_time > 0:
                gpu_events.append({
                    'name': event.key,
                    'gpu_time': gpu_time,
                    'count': event.count,
                    'avg_time': gpu_time / event.count if event.count > 0 else 0
                })
                total_gpu_time += gpu_time
                
                # Check if this event matches any matrix-vector patterns
                is_matvec = False
                for pattern in compiled_patterns:
                    if pattern.search(event.key):
                        matvec_events.append({
                            'name': event.key,
                            'gpu_time': gpu_time,
                            'count': event.count,
                            'avg_time': gpu_time / event.count if event.count > 0 else 0,
                            'pattern_matched': pattern.pattern
                        })
                        matvec_kernels[event.key].append(gpu_time)
                        total_matvec_time += gpu_time
                        is_matvec = True
                        
                        # Check if this is a new kernel
                        if event.key not in discovered_kernels:
                            new_kernels.append(event.key)
                            discovered_kernels.add(event.key)
                        
                        break
    
    # Calculate statistics
    matvec_percentage = (total_matvec_time / total_gpu_time * 100) if total_gpu_time > 0 else 0
    
    # Sort events by GPU time
    gpu_events.sort(key=lambda x: x['gpu_time'], reverse=True)
    matvec_events.sort(key=lambda x: x['gpu_time'], reverse=True)
    
    # Print analysis results
    logger.info(f"Context Length: {ctx_len}")
    logger.info(f"Total GPU Time: {total_gpu_time / 1000:.2f} ms")
    logger.info(f"Total Matrix-Vector Multiplication Time: {total_matvec_time / 1000:.2f} ms")
    logger.info(f"Matrix-Vector Multiplication Percentage: {matvec_percentage:.2f}%")
    logger.info(f"Number of Matrix-Vector Kernel Types: {len(matvec_kernels)}")
    logger.info(f"Total Matrix-Vector Events: {len(matvec_events)}")
    
    # Print only new kernels discovered in this run
    if new_kernels:
        logger.info("\n" + "="*50)
        logger.info("NEW MATRIX-VECTOR MULTIPLICATION KERNELS DISCOVERED")
        logger.info("="*50)
        for i, kernel_name in enumerate(new_kernels):
            logger.info(f"{i+1:2d}. {kernel_name}")
    else:
        logger.info("\nNo new matrix-vector multiplication kernels discovered in this run.")
    
    # Print total unique kernels discovered so far
    logger.info(f"\nTotal unique matrix-vector kernels discovered so far: {len(discovered_kernels)}")
    
    # Create pie chart for GPU kernel time distribution
    save_dir = f'../profiling/ctx_len_{ctx_len}'
    os.makedirs(save_dir, exist_ok=True)
    
    pie_chart_path = create_matvec_pie_chart(
        matvec_time_ms=total_matvec_time / 1000,
        total_time_ms=total_gpu_time / 1000,
        ctx_len=ctx_len,
        save_dir="./mat_vec_mul_inference_profiling"
    )
    
    # Save detailed results to JSON
    results = {
        'ctx_len': ctx_len,
        'total_gpu_time_ms': total_gpu_time / 1000,
        'total_matvec_time_ms': total_matvec_time / 1000,
        'matvec_percentage': matvec_percentage,
        'num_matvec_kernel_types': len(matvec_kernels),
        'num_matvec_events': len(matvec_events),
        'new_kernels_this_run': new_kernels,
        'total_unique_kernels_discovered': list(discovered_kernels),
        'top_gpu_events': gpu_events[:20],  # Top 20 events
        'matvec_events': matvec_events,
        'detection_patterns': matvec_patterns,
        'pie_chart_path': pie_chart_path
    }
    
    results_file = f'../profiling/ctx_len_{ctx_len}/matvec_analysis.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nDetailed analysis results saved to: {results_file}")
    logger.info("="*60)
    
    return results

def profile_attention_matmul(ctx_len):
    decode_tokens_to_generate = 10
    
    model_name = "meta-llama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    if ctx_len > model.config.max_position_embeddings:
        original_max_pos = model.config.max_position_embeddings
        model.config.max_position_embeddings = ctx_len + 1000
    
    inputs = get_ctx_len_text_from_wikitext_103_v1(ctx_len, tokenizer).to(model.device)
    attention_mask = torch.ones_like(inputs.input_ids)
    logger.info("Warming up the model...")
    with torch.no_grad():
        warmup_input = inputs.input_ids[:, :min(1024, inputs.input_ids.shape[1])]
        _ = model.generate(
            warmup_input, 
            max_new_tokens=1,
            do_sample=False,
            attention_mask=torch.ones_like(warmup_input),
            pad_token_id=tokenizer.eos_token_id
        )
    
    torch.cuda.synchronize()
    
    logger.info("Executing prefill phase (generating first token)...")
    
    with torch.no_grad():
        past_key_values = None
        input_ids = inputs.input_ids
            
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
            
        logits = outputs.logits[:, -1:, :]
        next_token_id = torch.argmax(logits, dim=-1)

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        past_key_values = outputs.past_key_values
    
    torch.cuda.synchronize()
    logger.info("Prefill phase completed, KV cache preserved")
    
    logger.info(f"Starting profiling for decode phase only (generating {decode_tokens_to_generate} tokens)...")

    profile_dir = f'../profiling/ctx_len_{ctx_len}'
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        with torch.no_grad():
            start_time = time.time()
            
            current_input_ids = input_ids
            current_past_key_values = past_key_values
            current_attention_mask = attention_mask
            
            progress_bar = tqdm(
                range(decode_tokens_to_generate), 
                desc=f"Generating tokens (ctx_len={ctx_len})",
                unit="token",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
            for i in progress_bar:
                decode_input_ids = current_input_ids[:, -1:]
                decode_attention_mask = torch.ones_like(decode_input_ids)
                        
                outputs = model(
                    input_ids=decode_input_ids,
                    attention_mask=decode_attention_mask,
                    past_key_values=current_past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                    
                logits = outputs.logits[:, -1:, :]
                next_token_id = torch.argmax(logits, dim=-1)
                    
                current_input_ids = torch.cat([current_input_ids, next_token_id], dim=-1)
                current_past_key_values = outputs.past_key_values
                current_attention_mask = torch.cat([current_attention_mask, torch.ones((current_attention_mask.shape[0], 1), device=current_attention_mask.device)], dim=-1)
                prof.step()
            
            torch.cuda.synchronize()
            end_time = time.time()
    
    generation_time = end_time - start_time
    new_tokens = current_input_ids.shape[1] - input_ids.shape[1]
    total_tokens = current_input_ids.shape[1] - inputs.input_ids.shape[1]
    
    logger.info(f"Decode phase: Generated {new_tokens} tokens in {generation_time:.3f} seconds")
    logger.info(f"Decode tokens per second: {new_tokens / generation_time:.2f}")
    logger.info(f"Total tokens generated: {total_tokens}")
    
    logger.info(f"TensorBoard logs automatically saved to: {profile_dir}")
    logger.info(f"To view in TensorBoard, run: tensorboard --logdir={profile_dir}")
    
    return prof

if __name__ == "__main__":
    ctx_len = 100 * 1024
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting profiling for context length: {ctx_len}")
    logger.info(f"{'='*60}")
        
    prof = profile_attention_matmul(ctx_len)
    logger.info(f"✅ Successfully completed profiling for context length: {ctx_len}")
    
    # Analyze matrix-vector multiplication kernels
    analysis_results = analyze_matvec_kernels(prof, ctx_len)
    logger.info(f"✅ Successfully completed matrix-vector multiplication analysis")
