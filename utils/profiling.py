from typing import Callable, Any, Optional
import os
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

def pytorch_profiling(enable_profiling: bool, save_path: Optional[str], func: Callable, *args, **kwargs) -> Any:
    if enable_profiling is False:
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            return *result, None
        return result, None

    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    func_name = func.__name__
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=tensorboard_trace_handler(os.path.join(save_path, func_name)) if save_path else None
    ) as prof:
        with record_function("model_inference"):
            result = func(*args, **kwargs)
    
    if isinstance(result, tuple):
        return *result, prof
    return result, prof