"""Version-robust KV-cache helpers and branch slicing."""

from typing import List, Tuple, Any
import torch
from transformers.cache_utils import DynamicCache


def kv_pairs(cache: Any) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract (key, value) pairs across layers from any HuggingFace Cache representation."""
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(l[0], l[1]) for l in cache]


def make_cache(pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    """Create a DynamicCache instance populated with the provided layer key/value tensors."""
    c = DynamicCache()
    for i, (k, v) in enumerate(pairs):
        c.update(k, v, i)
    return c


def slice_branch_kv(
    past_kv: DynamicCache,
    trunk_len: int,
    branch_start: int,
    branch_end: int,
    device: Any = None,
) -> DynamicCache:
    """
    Extract a contiguous KV slice containing the shared trunk tokens [0..trunk_len-1]
    and the branch tokens [branch_start..branch_end].
    """
    if device is None:
        pairs = kv_pairs(past_kv)
        device = pairs[0][0].device if pairs else "cpu"

    idx = torch.cat([
        torch.arange(0, trunk_len, device=device),
        torch.arange(branch_start, branch_end + 1, device=device),
    ])
    return make_cache([(k[:, :, idx, :].clone(), v[:, :, idx, :].clone()) for k, v in kv_pairs(past_kv)])


def crop_cache(cache: DynamicCache, keep: int) -> None:
    """Crop cache tokens back to length `keep` if it was extended."""
    extra = cache.get_seq_length() - keep
    if extra > 0:
        cache.crop(-extra)


def repeat_cache(cache: DynamicCache, n: int) -> DynamicCache:
    """Replicate batch dimension of KV cache by `n`."""
    return make_cache([(k.repeat(n, 1, 1, 1), v.repeat(n, 1, 1, 1)) for k, v in kv_pairs(cache)])


def select_batch_cache(cache: DynamicCache, batch_indices: torch.Tensor) -> DynamicCache:
    """Select a subset of rows along the batch dimension."""
    return make_cache([(k.index_select(0, batch_indices), v.index_select(0, batch_indices)) for k, v in kv_pairs(cache)])
