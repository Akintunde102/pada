"""Unit tests for tree sequence packing, 4D attention masks, and premise log-likelihood."""

import unittest
import torch
import torch.nn.functional as F

from pada.tree_engine.tree_builder import pack_guided_tree, compute_question_premise_ll


class MockTokenizer:

    def __init__(self):
        self.vocab = {f"tok_{i}": i for i in range(100)}

    def __len__(self):
        return 100

    def encode(self, text, add_special_tokens=False):
        # Deterministic token length based on word count
        words = text.strip().split()
        return list(range(len(words)))

    def decode(self, token_ids):
        return " ".join(f"tok_{i}" for i in token_ids)


class TestTreeBuilder(unittest.TestCase):

    def test_pack_guided_tree(self):
        tok = MockTokenizer()
        state = "A patient collapsed with acute pain."
        questions = {
            "q1": {"instructions": "What symptom occurred?"},
            "q2": {"instructions": "What is the severity?"},
        }
        id_to_word = {i: f"word_{i}" for i in range(100)}

        ids_t, mask_4d, pos_t, meta, S, seg = pack_guided_tree(
            state=state,
            questions=questions,
            tokenizer=tok,
            id_to_word=id_to_word,
            device="cpu",
            dtype=torch.float32,
        )

        total_len = ids_t.shape[1]
        self.assertEqual(ids_t.shape, (1, total_len))
        self.assertEqual(mask_4d.shape, (1, 1, total_len, total_len))
        self.assertEqual(pos_t.shape, (1, total_len))

        # Check metadata
        self.assertIn("q1", meta)
        self.assertIn("q2", meta)
        q1_meta = meta["q1"]
        q2_meta = meta["q2"]

        # Check restarted position IDs:
        # Trunk positions are 0..S-1
        self.assertTrue(torch.equal(pos_t[0, :S], torch.arange(S)))
        # Q1 starts at position S
        q1_len = q1_meta["branch_len"]
        self.assertEqual(pos_t[0, q1_meta["branch_start"]].item(), S)
        self.assertEqual(pos_t[0, q1_meta["branch_end"]].item(), S + q1_len - 1)
        # Q2 also starts at position S
        q2_len = q2_meta["branch_len"]
        self.assertEqual(pos_t[0, q2_meta["branch_start"]].item(), S)
        self.assertEqual(pos_t[0, q2_meta["branch_end"]].item(), S + q2_len - 1)

        # Check attention mask:
        # Branch 1 attending to Trunk: should be 0.0 (allowed)
        q1_idx = q1_meta["branch_start"]
        self.assertEqual(mask_4d[0, 0, q1_idx, 0].item(), 0.0)

        # Branch 1 attending to Branch 2: should be finfo.min (forbidden)
        q2_idx = q2_meta["branch_start"]
        self.assertEqual(mask_4d[0, 0, q1_idx, q2_idx].item(), torch.finfo(torch.float32).min)
        # Branch 2 attending to Branch 1: forbidden
        self.assertEqual(mask_4d[0, 0, q2_idx, q1_idx].item(), torch.finfo(torch.float32).min)

        # Causal mask: position i cannot attend to position i+1 within branch
        self.assertEqual(mask_4d[0, 0, q1_idx, q1_idx + 1].item(), torch.finfo(torch.float32).min)

    def test_compute_question_premise_ll(self):
        # 1 batch, 10 sequence length, 5 vocab
        vocab_size = 5
        seq_len = 10
        logits = torch.zeros((1, seq_len, vocab_size), dtype=torch.float32)

        # Set specific logit distributions
        # For pred_pos 3: token 2 has highest logit
        logits[0, 3, 2] = 10.0
        # For pred_pos 4: token 1 has highest logit
        logits[0, 4, 1] = 10.0

        meta = {
            "branch_start": 3,
            "branch_ids": [0, 2, 1, 4],  # branch tokens
            "premise_offsets": [1, 2],    # offsets inside branch
        }
        # offset 1: pred_pos = 3 + 1 - 1 = 3, target_tid = branch_ids[1] = 2
        # offset 2: pred_pos = 3 + 2 - 1 = 4, target_tid = branch_ids[2] = 1

        ll = compute_question_premise_ll(logits, meta)
        # Since target logits are 10.0 and others are 0.0, softmax probability ~ 1.0, logp ~ 0.0
        self.assertAlmostEqual(ll, 0.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
