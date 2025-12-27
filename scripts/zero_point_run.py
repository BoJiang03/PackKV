#! /usr/bin/env python
from typing import Dict, Tuple
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import accuracy_evaluation
from utils.config import PackKVCacheConfig
from utils.serialization import load, save
from utils.util import get_logger, block_other_logger, register_notify
from tqdm import tqdm
import multiprocessing

def run_accuracy_evaluation(config, benchmark, logger, result_queue):
    accuracy = accuracy_evaluation(config, benchmark, logger)
    result_queue.put(accuracy)

register_notify()

logger = get_logger(__file__)
block_other_logger(logger)
setting_path = "data/zero_point/zero_point_setting_map.pkl"
setting_map: Dict[int, Tuple[str, PackKVCacheConfig]] = load(setting_path)
logger.info(f"Setting map loaded from {setting_path} with {len(setting_map)} entries")

save_path = "data/zero_point/zero_point_result_map.pkl"
if os.path.exists(save_path):
    accuracy_result_map = load(save_path)
else:
    accuracy_result_map = {}

result_queue = multiprocessing.Queue()

for hash, pair in tqdm(setting_map.items(), total=len(setting_map), desc="Evaluating accuracy"):
    if hash in accuracy_result_map:
        logger.info(f"Skip: {hash}")
        continue
    benchmark, config = pair
    process = multiprocessing.Process(
        target=run_accuracy_evaluation,
        args=(config, benchmark, logger, result_queue)
    )
    process.start()
    logger.info("evaluation process started")
    process.join(3600) # seconds
    if process.is_alive():
        logger.warning(f"Process for {hash} timed out. Terminating.")
        process.terminate()
        process.join()
        accuracy = {}
    else:
        logger.info("evaluation process joined")
        accuracy = result_queue.get()
    accuracy_result_map[hash] = accuracy
    save(accuracy_result_map, save_path)
    logger.info(f"\n{hash}\n{config}\n{accuracy}\n\n")