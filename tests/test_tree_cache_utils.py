"""Unit tests for KV cache manipulation utilities."""

import unittest
import torch
from transformers.cache_utils import DynamicCache

from pada.tree_engine.cache_utils import (
    kv_pairs,
    make_cache,
    crop_cache,
    repeat_cache,
    select_batch_cache,
    slice_branch_kv,
)


class TestCacheUtils(unittest.TestCase):

    def setUp(self):
        # Create a mock 2-layer cache with batch=1, heads=2, seq_len=10, head_dim=4
        self.k1 = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1).expand(1, 2, 10, 4).clone()
        self.v1 = self.k1 * 10
        self.k2 = self.k1 + 100
        self.v2 = self.v1 + 100

        self.cache = DynamicCache()
        self.cache.update(self.k1, self.v1, 0)
        self.cache.update(self.k2, self.v2, 1)

    def test_kv_pairs_and_make_cache(self):
        pairs = kv_pairs(self.cache)
        self.assertEqual(len(pairs), 2)
        k, v = pairs[0]
        self.assertEqual(k.shape, (1, 2, 10, 4))
        self.assertEqual(v.shape, (1, 2, 10, 4))

        new_cache = make_cache(pairs)
        self.assertEqual(new_cache.get_seq_length(), 10)
        new_pairs = kv_pairs(new_cache)
        self.assertTrue(torch.equal(new_pairs[0][0], self.k1))

    def test_crop_cache(self):
        crop_cache(self.cache, keep=6)
        self.assertEqual(self.cache.get_seq_length(), 6)
        pairs = kv_pairs(self.cache)
        self.assertEqual(pairs[0][0].shape[-2], 6)

    def test_repeat_cache(self):
        repeated = repeat_cache(self.cache, n=3)
        self.assertEqual(repeated.get_seq_length(), 10)
        pairs = kv_pairs(repeated)
        self.assertEqual(pairs[0][0].shape[0], 3)
        self.assertTrue(torch.equal(pairs[0][0][0], pairs[0][0][1]))

    def test_select_batch_cache(self):
        repeated = repeat_cache(self.cache, n=4)
        indices = torch.tensor([1, 3], dtype=torch.long)
        selected = select_batch_cache(repeated, indices)
        self.assertEqual(selected.get_seq_length(), 10)
        pairs = kv_pairs(selected)
        self.assertEqual(pairs[0][0].shape[0], 2)

    def test_slice_branch_kv(self):
        # Suppose trunk_len = 4 (indices 0, 1, 2, 3), branch is [6..8] (indices 6, 7, 8)
        # Expected sliced length = 4 + 3 = 7
        sliced = slice_branch_kv(self.cache, trunk_len=4, branch_start=6, branch_end=8, device="cpu")
        self.assertEqual(sliced.get_seq_length(), 7)

        pairs = kv_pairs(sliced)
        k_sliced = pairs[0][0]  # shape (1, 2, 7, 4)
        expected_indices = torch.tensor([0, 1, 2, 3, 6, 7, 8])
        expected_k = self.k1[:, :, expected_indices, :]
        self.assertTrue(torch.equal(k_sliced, expected_k))


if __name__ == "__main__":
    unittest.main()
