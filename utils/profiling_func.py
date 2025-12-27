import textwrap
from typing import Optional, Dict, List, Tuple

import torch
from torch.profiler import profile, ProfilerActivity


def profile_func(func, warmup_runs: int = 5, profile_runs: int = 3,
                 tensorboard_dir: Optional[str] = None) -> Tuple[Dict[str, List], float]:
    print(f"Starting warmup with {warmup_runs} runs...")

    for i in range(warmup_runs):
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"Warmup run {i + 1}/{warmup_runs} completed")

    print(f"\nStarting profiling with {profile_runs} runs...")

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Cannot profile GPU kernels.")
        return {
            'kernel_names': [],
            'kernel_times': [],
            'total_time': 0.0
        }, 0.0

    activities = [ProfilerActivity.CUDA]

    # Use CUDA events to measure per-run time while profiling
    times_ms: List[float] = []

    with profile(
            activities=activities,
            record_shapes=True,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_dir) if tensorboard_dir else None,
    ) as prof:
        for i in range(profile_runs):
            if torch.cuda.is_available():
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)

                start_evt.record()
                func()
                end_evt.record()
                torch.cuda.synchronize()

                # elapsed_time: milliseconds
                times_ms.append(start_evt.elapsed_time(end_evt))
            else:
                func()

            prof.step()
            print(f"Profile run {i + 1}/{profile_runs} completed")

    kernel_info = {
        'kernel_names': [],
        'kernel_times': [],  # average per-run time in microseconds
        'total_time': 0.0    # average per-run total time in microseconds
    }

    events = prof.key_averages()

    for event in events:
        if event.device_type == torch.profiler.DeviceType.CUDA:
            # Use self_cuda_time_total for CUDA kernel time (aggregated over all profile runs)
            cuda_time_total = event.self_cuda_time_total if hasattr(
                event, 'self_cuda_time_total'
            ) else event.self_device_time_total

            # Convert to per-run average by dividing by number of profile runs
            cuda_time_avg = cuda_time_total / profile_runs if profile_runs > 0 else 0.0

            kernel_info['kernel_names'].append(event.key)
            kernel_info['kernel_times'].append(cuda_time_avg)
            kernel_info['total_time'] += cuda_time_avg

    print("\n" + "=" * 80)
    print("GPU Kernel Profiling Summary")
    print("=" * 80)
    print(f"Total unique GPU kernels: {len(kernel_info['kernel_names'])}")
    print(f"Total GPU time: {kernel_info['total_time']:.2f} us ({kernel_info['total_time'] / 1000:.2f} ms)")

    if len(kernel_info['kernel_names']) > 0:
        sorted_kernels = sorted(zip(kernel_info['kernel_names'], kernel_info['kernel_times']),
                                key=lambda x: x[1], reverse=True)

        # Calculate coverage of top kernels
        top_20_time = sum(time for _, time in sorted_kernels[:20])
        top_20_coverage = (top_20_time / kernel_info['total_time'] * 100) if kernel_info['total_time'] > 0 else 0

        print(f"\nTop 20 GPU kernels by time (covering {top_20_coverage:.1f}% of total GPU time):")
        print("=" * 80)

        for i, (name, time_us) in enumerate(sorted_kernels[:20], 1):
            percentage = (time_us / kernel_info['total_time'] * 100) if kernel_info['total_time'] > 0 else 0
            time_str = f"{time_us:>12.2f} us ({percentage:>5.1f}%)"

            # Top 10: show full name with multi-line support
            if i <= 10:
                print()  # Empty line before each kernel
                # Wrap long kernel names, reserving space for time info on the right
                # Total width: 80, reserve 30 for time info, 4 for "XX. ", leaves 46 for name
                max_first_line_width = 46

                if len(name) <= max_first_line_width:
                    # Short name: display in one line with time on the right
                    print(f"{i:2d}. {name:<{max_first_line_width}} {time_str}")
                else:
                    # Long name: first line with time on the right, continuation lines below
                    wrapped_lines = textwrap.wrap(name, width=max_first_line_width)
                    if wrapped_lines:
                        # First line with time info on the right
                        print(f"{i:2d}. {wrapped_lines[0]:<{max_first_line_width}} {time_str}")
                        # Continuation lines (indented, no time info)
                        for line in wrapped_lines[1:]:
                            print(f"    {line}")
            # 11-20: show truncated name in single line
            else:
                truncated_name = name[:60] if len(name) > 60 else name
                print(f"{i:2d}. {truncated_name:<60} {time_us:>12.2f} us ({percentage:>5.1f}%)")

        if len(sorted_kernels) > 20:
            remaining_kernels = len(sorted_kernels) - 20
            remaining_time = kernel_info['total_time'] - top_20_time
            remaining_pct = 100.0 - top_20_coverage
            print("-" * 80)
            print(
                f"... and {remaining_kernels} more kernels accounting for {remaining_time:.2f} us ({remaining_pct:.1f}%)")

    if tensorboard_dir:
        print(f"\nTensorBoard trace exported to: {tensorboard_dir}")
        print(f"View with: tensorboard --logdir={tensorboard_dir}")

    print("=" * 80)

    # Average func wall-clock time per run (in milliseconds, from CUDA events)
    avg_time = (sum(times_ms) / len(times_ms)) if times_ms else 0.0

    return kernel_info, avg_time