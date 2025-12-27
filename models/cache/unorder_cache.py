from typing import Optional, Dict, Any, Tuple

import torch
from transformers import Cache

from utils.compute import safe_cat

class UnorderCacheStatic:
    enable_unorder_cache = True

def random_order(k, v):
    if k is None or v is None:
        return k, v
    
    seq_len = k.shape[2]
    perm_indices = torch.randperm(seq_len, device=k.device)
    k_reordered = k[:, :, perm_indices, :]
    v_reordered = v[:, :, perm_indices, :]
    
    return k_reordered, v_reordered

from transformers.utils.logging import get_logger

logger = get_logger(__file__)

class UnorderCache(Cache):
    def __init__(self):
        super().__init__()
        self.k_cache = [None] * 32
        self.v_cache = [None] * 32
        logger.warning_once("\nwarning_once: UnorderCache")

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.k_cache[layer_idx] = safe_cat(self.k_cache[layer_idx], key_states, dim=2)
        self.v_cache[layer_idx] = safe_cat(self.v_cache[layer_idx], value_states, dim=2)
        is_prefill = key_states.shape[2] != 1
        if is_prefill:
            if layer_idx == 0:
                print("\nUnorderCache.update called.", flush=True)
            return self.k_cache[layer_idx], self.v_cache[layer_idx]
        else:
            return random_order(self.k_cache[layer_idx], self.v_cache[layer_idx])

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        first_k_cache = self.k_cache[layer_idx]
        return 0 if first_k_cache is None else first_k_cache.shape[2]