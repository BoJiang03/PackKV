#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import throughput_evaluation
from utils.config import PackKVCacheConfig
from utils.serialization import load, save
from utils.util import get_logger, block_other_logger, register_notify
from utils.gpu_scheduler import run_multi_gpu_tasks
import torch



def run_throughput_evaluation_wrapper(config, ctx_len, logger, hash_key):
    """
    Wrapper function for throughput evaluation that doesn't need result_queue.
    """
    result = throughput_evaluation(config, ctx_len, enable_save=False, logger=logger)
    return result



# register_notify()

if __name__ == '__main__':
    logger = get_logger(__file__)
    block_other_logger(logger)
    
    # Auto-detect GPU count and create tasks_devices_map
    gpu_count = torch.cuda.device_count()
    logger.info(f"Detected {gpu_count} GPUs")

    # Create tasks_devices_map based on available GPUs
    tasks_devices_map = [(0, 1, 2, 3)]
    logger.info(f"Tasks devices map: {tasks_devices_map}")
    
    setting_path = "data/throughput/throughput_setting_map_a100.pkl"
    setting_map: Dict[int, Tuple[int, PackKVCacheConfig]] = load(setting_path)
    logger.info(f"Setting map loaded from {setting_path} with {len(setting_map)} entries")

    save_path = "data/throughput/throughput_result_map_a100.pkl"
    if os.path.exists(save_path):
        throughput_result_map = load(save_path)
    else:
        throughput_result_map = {}

    def process_result_with_save(args, result):
        """
        After process function to handle results and save them.
        
        Args:
            args: The original arguments (config, ctx_len, logger, hash_key)
            result: The result from run_throughput_evaluation_wrapper
        
        Returns:
            Tuple of (hash_key, result) for later processing
        """
        config, ctx_len, logger, hash_key = args
        
        # Process and save result
        if result is not None and not isinstance(result, str):
            throughput_result_map[hash_key] = result
            logger.info(f"\n{hash_key}\nResult: {result}\n")
        else:
            logger.warning(f"Failed to get result for {hash_key}")
            logger.warning(f"Result: {result}")
            # throughput_result_map[hash_key] = {}
        
        # Save the updated result map after each task completion
        save(throughput_result_map, save_path)
        logger.info(f"Saved results to {save_path}")
        
        return (hash_key, result)

    # Prepare tasks for GPU scheduler
    tasks_to_run = []
    for hash_key, pair in setting_map.items():
        if hash_key in throughput_result_map:
            logger.info(f"Skip: {hash_key}")
            continue
        ctx_len, config = pair
        # Create task arguments tuple
        task_args = (config, ctx_len, logger, hash_key)
        tasks_to_run.append(task_args)

    logger.info(f"Total tasks to run: {len(tasks_to_run)}")

    if tasks_to_run:
        # Run tasks using GPU scheduler
        results = run_multi_gpu_tasks(
            tasks_devices_map=tasks_devices_map,
            target_func=run_throughput_evaluation_wrapper,
            args_list=tasks_to_run,
            after_process_func=process_result_with_save,
            timeout=3600,
            logger=logger
        )
        
        logger.info(f"All tasks completed successfully. Final results saved to {save_path}")
    else:
        logger.info("No new tasks to run. All tasks already completed.") 