import torch
import multiprocessing as mp
import queue
import threading
import time
import os
import logging
from pathlib import Path
from typing import List, Tuple, Callable, Any, Dict
from utils.util import get_logger
from tqdm import tqdm

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# This must be done before any other multiprocessing operations
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

def _standalone_worker_wrapper(device_mapping: tuple, target_func: Callable, args: Tuple, result_queue: mp.Queue):
    """
    Standalone worker function that sets CUDA_VISIBLE_DEVICES and calls the target function.
    This function is outside the class to avoid pickling issues with spawn.
    Supports both single and multi-GPU assignments.
    """
    # Set CUDA_VISIBLE_DEVICES to show the assigned GPU(s)
    if len(device_mapping) == 1:
        # Single GPU case
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_mapping[0])
    else:
        # Multi-GPU case: comma-separated list
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_mapping))
    
    # Initialize CUDA in spawn mode (clean slate)
    import torch
    if torch.cuda.is_available():
        # In spawn mode, we start with a clean CUDA state
        # Just verify that the expected number of devices are visible
        num_visible_devices = torch.cuda.device_count()
        expected_devices = len(device_mapping)
        
        if num_visible_devices != expected_devices:
            raise RuntimeError(f"Expected {expected_devices} CUDA devices but found {num_visible_devices}")
        
        # Set default device to 0 (first remapped device)
        torch.cuda.set_device(0)
        

    
    try:
        # Call the target function
        result = target_func(*args)
        result_queue.put(('success', args, result))
    except Exception as e:
        result_queue.put(('error', args, str(e)))

class GPUTaskScheduler:
    """
    A GPU task scheduler that automatically manages multiple GPUs and keeps them busy with tasks.
    """
    
    def __init__(self, tasks_devices_map: List[Tuple], timeout: int = 3600, logger=None):
        """
        Initialize the GPU task scheduler.
        
        Args:
            tasks_devices_map: List of tuples representing GPU device mappings
            timeout: Timeout for each process in seconds
            logger: Logger instance
        """
        self.tasks_devices_map = tasks_devices_map
        self.timeout = timeout
        self.logger = logger or get_logger(__name__)
        
        self.gpu_count = torch.cuda.device_count()
            
        self.logger.info(f"Detected {self.gpu_count} GPUs")
        
        # Validate tasks_devices_map
        max_device = max(device[0] if isinstance(device, tuple) else device for device in tasks_devices_map)
        if max_device >= self.gpu_count:
            raise ValueError(f"Device {max_device} specified in tasks_devices_map but only {self.gpu_count} GPUs available")
        
        # Create dedicated loggers for each task queue
        self.queue_loggers = self._create_queue_loggers()
        
        # Initialize queues and tracking structures
        self.task_queues = [queue.Queue() for _ in range(len(tasks_devices_map))]
        self.result_queue = mp.Queue()
        self.active_processes = {}  # device_idx -> (process, start_time, task_args)
        self.completed_results = []
        self.pending_results = []  # Store results that need processing
        self.total_tasks = 0
        self.completed_tasks = 0
        self.last_logged_progress = -1  # Track last logged progress to avoid spam
        self.progress_bar = None  # Progress bar instance
        
    def _create_queue_loggers(self) -> List[logging.Logger]:
        """
        Create dedicated loggers for each task queue based on the main logger's file path.
        
        Returns:
            List of loggers, one for each task queue
        """
        queue_loggers = []
        
        # Try to find the file handler from the main logger
        log_file_path = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_path = handler.baseFilename
                break
        
        if log_file_path is None:
            # If no file handler found, create loggers without file output
            for i, device_mapping in enumerate(self.tasks_devices_map):
                logger_name = f"GPUScheduler_queue_{i}_{id(self)}"  # Add unique ID to avoid conflicts
                queue_logger = logging.getLogger(logger_name)
                queue_logger.setLevel(self.logger.level)
                # Prevent propagation to parent logger
                queue_logger.propagate = False
                queue_loggers.append(queue_logger)
            return queue_loggers
        
        # Extract path components
        log_path = Path(log_file_path)
        log_dir = log_path.parent
        log_stem = log_path.stem  # filename without extension
        log_suffix = log_path.suffix  # file extension
        
        # Log the source file path for debugging
        self.logger.info(f"Base log file: {log_file_path}")
        self.logger.info(f"Will create GPU-specific log files in: {log_dir}")
        
        # Create logger for each task queue
        for i, device_mapping in enumerate(self.tasks_devices_map):
            # Create descriptive filename based on GPU devices
            if isinstance(device_mapping, tuple):
                if len(device_mapping) == 1:
                    gpu_desc = f"gpu{device_mapping[0]}"
                else:
                    gpu_desc = f"gpu{'_'.join(map(str, device_mapping))}"
            else:
                gpu_desc = f"gpu{device_mapping}"
            
            # Create new log filename: xx/xx/xxx.log -> xx/xx/xxx_gpu0.log
            new_log_file = log_dir / f"{log_stem}_{gpu_desc}{log_suffix}"
            
            # Ensure directory exists
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create logger with unique name to avoid conflicts
            logger_name = f"GPUScheduler_queue_{i}_{gpu_desc}_{id(self)}"
            queue_logger = logging.getLogger(logger_name)
            queue_logger.setLevel(self.logger.level)
            
            # CRITICAL: Prevent propagation to parent logger
            # This ensures logs go only to our file handler, not to parent loggers
            queue_logger.propagate = False
            
            # Clear any existing handlers to avoid duplicates
            queue_logger.handlers.clear()
            
            # Create file handler
            file_handler = logging.FileHandler(new_log_file)
            file_handler.setLevel(self.logger.level)
            
            # Copy formatter from main logger if available
            formatter = None
            for handler in self.logger.handlers:
                if hasattr(handler, 'formatter') and handler.formatter:
                    formatter = handler.formatter
                    break
            
            if formatter:
                file_handler.setFormatter(formatter)
            else:
                # Default formatter
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(formatter)
            
            queue_logger.addHandler(file_handler)
            queue_loggers.append(queue_logger)
            
            # Test the logger to ensure it's working
            queue_logger.info(f"Queue logger initialized for {gpu_desc}")
            # Force flush to ensure immediate write
            file_handler.flush()
            
            self.logger.info(f"Created queue logger for {gpu_desc}: {new_log_file}")
        
        return queue_loggers
        
    def _get_device_id(self, device_idx: int) -> int:
        """Get the first GPU device ID from tasks_devices_map (for backward compatibility)."""
        device_mapping = self.tasks_devices_map[device_idx]
        return device_mapping[0] if isinstance(device_mapping, tuple) else device_mapping
    
    def _get_device_mapping(self, device_idx: int):
        """Get the complete GPU device mapping from tasks_devices_map."""
        device_mapping = self.tasks_devices_map[device_idx]
        if isinstance(device_mapping, tuple):
            return device_mapping
        else:
            return (device_mapping,)
    
    def _get_queue_logger(self, device_idx: int) -> logging.Logger:
        """Get the dedicated logger for a specific device queue."""
        if 0 <= device_idx < len(self.queue_loggers):
            return self.queue_loggers[device_idx]
        return self.logger  # Fallback to main logger
    
    def _flush_queue_logger(self, device_idx: int):
        """Force flush the queue logger's file handlers to ensure immediate write."""
        if 0 <= device_idx < len(self.queue_loggers):
            queue_logger = self.queue_loggers[device_idx]
            for handler in queue_logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
    
    def _flush_all_queue_loggers(self):
        """Force flush all queue loggers' file handlers."""
        for queue_logger in self.queue_loggers:
            for handler in queue_logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
    
    def _init_progress_bar(self, total_tasks: int):
        """Initialize the progress bar."""
        self.progress_bar = tqdm(
            total=total_tasks,
            desc="GPU Tasks",
            unit="task",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    
    def _update_progress_bar(self, completed_count: int = 1):
        """Update the progress bar."""
        if self.progress_bar is not None:
            self.progress_bar.update(completed_count)
    
    def _close_progress_bar(self):
        """Close and cleanup the progress bar."""
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
    

    
    def _log_progress_if_needed(self):
        """
        Log progress only when it reaches certain milestones and has actually changed.
        """
        # Define progress milestones
        milestones = [0]
        step = max(1, self.total_tasks // 10)  # Log every 10% or at least every 1 task
        for i in range(step, self.total_tasks, step):
            milestones.append(i)
        milestones.append(self.total_tasks)
        
        # Only log if we've reached a milestone and haven't logged this progress yet
        if (self.completed_tasks in milestones and 
            self.completed_tasks != self.last_logged_progress):
            self.logger.info(f"Progress: {self.completed_tasks}/{self.total_tasks} tasks completed")
            self.last_logged_progress = self.completed_tasks
    
    def _process_all_results_serially(self, after_process_func: Callable = None):
        """
        Process all pending results in a COMPLETELY SERIAL manner.
        This ensures after_process_func is never called concurrently.
        
        IMPORTANT: This method guarantees that after_process_func is called
        one at a time, in sequence, with no threading concerns.
        """
        if not after_process_func:
            # If no processing function, just add raw results
            self.completed_results.extend([result for _, result in self.pending_results])
        else:
            # Process each result one by one, serially
            for args, result in self.pending_results:
                processed_result = after_process_func(args, result)
                self.completed_results.append(processed_result)
        
        # Clear processed results
        self.pending_results.clear()
    
    def _monitor_processes_once(self, after_process_func: Callable = None):
        """
        Monitor active processes and handle completed/timed-out tasks (single iteration).
        """
        current_time = time.time()
        completed_devices = []
        
        # Check for completed processes
        for device_idx, (process, start_time, task_args) in self.active_processes.items():
            queue_logger = self._get_queue_logger(device_idx)
            
            if not process.is_alive():
                # Process completed
                completed_devices.append(device_idx)
                # Only log to queue logger for detailed tracking
                device_mapping = self._get_device_mapping(device_idx)
                device_str = f"device {device_mapping[0]}" if len(device_mapping) == 1 else f"devices {device_mapping}"
                queue_logger.info(f"Process on {device_str} completed")
                # Force flush to ensure immediate write
                self._flush_queue_logger(device_idx)
            elif current_time - start_time > self.timeout:
                # Process timed out
                device_mapping = self._get_device_mapping(device_idx)
                device_str = f"device {device_mapping[0]}" if len(device_mapping) == 1 else f"devices {device_mapping}"
                queue_logger.warning(f"Process on {device_str} timed out. Terminating.")
                # Also log timeout to main logger as it's important
                self.logger.warning(f"Process timeout on {device_str}")
                # Force flush to ensure immediate write
                self._flush_queue_logger(device_idx)
                process.terminate()
                process.join()
                completed_devices.append(device_idx)
                # Add timeout result to pending
                self.pending_results.append((task_args, None))
        
        # Handle completed processes
        for device_idx in completed_devices:
            process, start_time, task_args = self.active_processes.pop(device_idx)
            process.join()  # Ensure process is cleaned up
            self.completed_tasks += 1
        
        # Update progress bar for completed tasks
        if completed_devices:
            self._update_progress_bar(len(completed_devices))
        
        # Log progress only at milestones and when it actually changes
        self._log_progress_if_needed()
        
        # Collect results from the queue (but don't process them yet)
        while True:
            try:
                status, result_args, result = self.result_queue.get_nowait()
                self.pending_results.append((result_args, result))
            except queue.Empty:
                break
        
        # Process ALL pending results in a single, serial operation
        # This guarantees no concurrent calls to after_process_func
        if self.pending_results:
            self._process_all_results_serially(after_process_func)
        
        # Start new processes on available devices
        for device_idx in range(len(self.tasks_devices_map)):
            if device_idx not in self.active_processes and not self.task_queues[device_idx].empty():
                try:
                    target_func, task_args = self.task_queues[device_idx].get_nowait()
                    device_mapping = self._get_device_mapping(device_idx)
                    queue_logger = self._get_queue_logger(device_idx)
                    
                    # Start new process
                    process = mp.Process(
                        target=_standalone_worker_wrapper,
                        args=(device_mapping, target_func, task_args, self.result_queue)
                    )
                    process.start()
                    self.active_processes[device_idx] = (process, time.time(), task_args)
                    
                    # Only log to queue logger for detailed tracking
                    device_str = f"device {device_mapping[0]}" if len(device_mapping) == 1 else f"devices {device_mapping}"
                    queue_logger.info(f"Started process on {device_str} for task")
                    # Force flush to ensure immediate write
                    self._flush_queue_logger(device_idx)
                    
                except queue.Empty:
                    continue
    
    def run_tasks(self, target_func: Callable, args_list: List[Tuple], after_process_func: Callable = None) -> List[Any]:
        """
        Run tasks across multiple GPUs with automatic load balancing and progress bar.
        
        IMPORTANT: The after_process_func is guaranteed to be called SERIALLY.
        You do not need to worry about threading or concurrent access when
        writing your after_process_func - it will be called one at a time.
        
        PROGRESS BAR: A visual progress bar will be displayed showing:
        - Current progress (completed/total tasks)
        - Elapsed time and estimated remaining time
        - Processing rate (tasks per second)
        
        Args:
            target_func: The function to run for each task
            args_list: List of argument tuples for each task
            after_process_func: Optional function to process results (args, result) -> processed_result
                               This function is called SERIALLY - no threading concerns!
            
        Returns:
            List of results from all tasks
        """
        self.total_tasks = len(args_list)
        self.completed_tasks = 0
        self.completed_results = []
        self.pending_results = []
        self.last_logged_progress = -1
        
        # Initialize progress bar
        self._init_progress_bar(self.total_tasks)
        
        # Distribute tasks across device queues in round-robin fashion
        for i, args in enumerate(args_list):
            device_idx = i % len(self.tasks_devices_map)
            self.task_queues[device_idx].put((target_func, args))
        
        self.logger.info(f"Distributed {len(args_list)} tasks across {len(self.tasks_devices_map)} device queues")
        
        # Log task distribution for each queue (only to queue loggers)
        for device_idx in range(len(self.tasks_devices_map)):
            queue_size = self.task_queues[device_idx].qsize()
            if queue_size > 0:
                queue_logger = self._get_queue_logger(device_idx)
                device_mapping = self.tasks_devices_map[device_idx]
                queue_logger.info(f"Queue for devices {device_mapping} has {queue_size} tasks assigned")
                # Force flush to ensure immediate write
                self._flush_queue_logger(device_idx)
        
        # Start initial processes (one per device if tasks available)
        active_devices = []
        for device_idx in range(len(self.tasks_devices_map)):
            if not self.task_queues[device_idx].empty():
                try:
                    target_func, task_args = self.task_queues[device_idx].get_nowait()
                    device_mapping = self._get_device_mapping(device_idx)
                    queue_logger = self._get_queue_logger(device_idx)
                    
                    process = mp.Process(
                        target=_standalone_worker_wrapper,
                        args=(device_mapping, target_func, task_args, self.result_queue)
                    )
                    process.start()
                    self.active_processes[device_idx] = (process, time.time(), task_args)
                    
                    # Log to queue logger for detailed tracking
                    device_str = f"device {device_mapping[0]}" if len(device_mapping) == 1 else f"devices {device_mapping}"
                    queue_logger.info(f"Started initial process on {device_str}")
                    # Force flush to ensure immediate write
                    self._flush_queue_logger(device_idx)
                    active_devices.extend(device_mapping)
                    
                except queue.Empty:
                    continue
        
        # Log summary to main logger
        if active_devices:
            self.logger.info(f"Started initial processes on {len(active_devices)} GPUs: {active_devices}")
        
        # Log initial progress
        self._log_progress_if_needed()
        
        try:
            # Main monitoring loop - ALL after_process_func calls happen in THIS thread only
            while self.completed_tasks < self.total_tasks:
                self._monitor_processes_once(after_process_func)
                time.sleep(0.1)
            
            # Clean up any remaining processes
            for device_idx, (process, start_time, task_args) in list(self.active_processes.items()):
                if process.is_alive():
                    process.terminate()
                    process.join()
            
            self.logger.info(f"All {self.total_tasks} tasks completed successfully")
            
        except KeyboardInterrupt:
            self.logger.info("Task execution interrupted by user")
            # Clean up any remaining processes
            for device_idx, (process, start_time, task_args) in list(self.active_processes.items()):
                if process.is_alive():
                    process.terminate()
                    process.join()
            raise
        except Exception as e:
            self.logger.error(f"Error during task execution: {str(e)}")
            # Clean up any remaining processes
            for device_idx, (process, start_time, task_args) in list(self.active_processes.items()):
                if process.is_alive():
                    process.terminate()
                    process.join()
            raise
        finally:
            # Always close progress bar
            self._close_progress_bar()
        
        # Log completion summary for each queue (only to queue loggers)
        for device_idx, queue_logger in enumerate(self.queue_loggers):
            device_mapping = self.tasks_devices_map[device_idx]
            queue_logger.info(f"All tasks completed for devices {device_mapping}")
            # Force flush to ensure immediate write
            self._flush_queue_logger(device_idx)
        
        # Final flush of all queue loggers to ensure all logs are written
        self._flush_all_queue_loggers()
        
        return self.completed_results


def run_multi_gpu_tasks(tasks_devices_map: List[Tuple], 
                       target_func: Callable, 
                       args_list: List[Tuple], 
                       after_process_func: Callable = None,
                       timeout: int = 3600,
                       logger=None) -> List[Any]:
    """
    Convenient function to run tasks across multiple GPUs with progress bar.
    
    THREAD-SAFETY GUARANTEE: The after_process_func is called SERIALLY.
    You can safely access shared resources in after_process_func without
    worrying about race conditions or concurrent access.
    
    PROGRESS BAR: A visual progress bar will be displayed showing:
    - Current progress (completed/total tasks)
    - Elapsed time and estimated remaining time
    - Processing rate (tasks per second)
    
    Args:
        tasks_devices_map: List of tuples representing GPU device mappings
        target_func: The function to run for each task
        args_list: List of argument tuples for each task  
        after_process_func: Optional function to process results (args, result) -> processed_result
                           GUARANTEED to be called serially - no threading concerns!
        timeout: Timeout for each process in seconds
        logger: Logger instance (dedicated loggers will be created for each GPU queue)
        
    Returns:
        List of results from all tasks
    """
    scheduler = GPUTaskScheduler(tasks_devices_map, timeout, logger)
    return scheduler.run_tasks(target_func, args_list, after_process_func) 