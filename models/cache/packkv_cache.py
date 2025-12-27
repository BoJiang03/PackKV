from typing import Optional, Dict, Any, Tuple

import torch
from transformers import Cache
import os

from models.cache.packkv_quant import PackKVCachePytorchQuant

def save_to_file(path, tensor, layer_idx, is_k):
    # check path
    path = os.path.join(path, "k" if is_k else "v")
    if not os.path.exists(path):
        os.makedirs(path)
    tensor_name = f"{layer_idx}_{'k' if is_k else 'v'}.pt"
    # check file exists
    file_path = os.path.join(path, tensor_name)
    if os.path.exists(file_path):
        print(f"File {tensor_name} already exists")
        exit()
    torch.save(tensor, file_path)

class PackKVCache(Cache):
    def __init__(self, batch_size, head_num, head_dim, layer_num):
        super().__init__()
        # self.cache_instance = KVCompCacheCuda(batch_size, head_num, head_dim, layer_num)
        self.cache_instance = PackKVCachePytorchQuant(batch_size, head_num, head_dim, layer_num)

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cache_instance.update(key_states, value_states, layer_idx, cache_kwargs)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.cache_instance.get_seq_length(layer_idx)
