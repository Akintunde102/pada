"""Integration tests for TreeEngine end-to-end execution with a lightweight mock model."""

import unittest
from types import SimpleNamespace
import torch
from transformers.cache_utils import DynamicCache

from pada.tree_engine import (
    TreeEngine,
    init_engine,
    get_engine,
    predict_unified_tree,
    precision,
    model_dtype,
)


class MockIntegrationModel(torch.nn.Module):
    """Mock causal LM producing deterministic logits and handling past_key_values."""

    def __init__(self, vocab_size=50):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.linear = torch.nn.Linear(vocab_size, vocab_size, bias=False)
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape
        # Create deterministic logits where token 5 and 6 have prominent values
        logits = torch.zeros((batch_size, seq_len, self.vocab_size), dtype=self.weight.dtype)
        logits[:, :, 5] = 4.0
        logits[:, :, 6] = 3.0
        logits[:, :, 0] = 5.0  # Stop token

        cache = past_key_values
        if use_cache:
            if cache is None:
                cache = DynamicCache()
            k = torch.zeros((batch_size, 1, seq_len, 4), dtype=self.weight.dtype)
            v = torch.zeros((batch_size, 1, seq_len, 4), dtype=self.weight.dtype)
            cache.update(k, v, 0)

        return SimpleNamespace(logits=logits, past_key_values=cache)


class MockIntegrationTokenizer:

    def __init__(self, vocab_size=50):
        self.vocab_size = vocab_size
        self.vocab = {
            "<|endoftext|>": 0,
            "<|im_end|>": 1,
            " ": 2,
            "Question:": 3,
            " One-word answer:": 4,
            " cardiology": 5,
            " critical": 6,
            " hosp": 7,
            "ital": 8,
        }
        for i in range(len(self.vocab), vocab_size):
            self.vocab[f"word_{i}"] = i
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        # Map known prefixes / phrases
        if "One-word answer:" in text:
            return [3, 5, 4]
        return [2, 5]

    def decode(self, token_ids):
        return "".join(self.inv_vocab.get(t, f"tok_{t}") for t in token_ids)

    def batch_decode(self, sequences, clean_up_tokenization_spaces=False):
        return [self.decode(s) for s in sequences]

    def convert_ids_to_tokens(self, tid):
        return self.inv_vocab.get(int(tid), f"tok_{tid}")


class TestTreeEngineIntegration(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 30
        self.model = MockIntegrationModel(vocab_size=self.vocab_size)
        self.tokenizer = MockIntegrationTokenizer(vocab_size=self.vocab_size)
        self.engine = TreeEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu",
            dtype=torch.float32,
            diag_mode="all",
        )

    def test_predict_constrained_and_open(self):
        situation = "Patient presented with severe chest pain."
        questions = {
            "specialty": {
                "instructions": "Which medical specialty handles this?",
                "candidates": ["cardiology", "neurology"],
                "expected": "cardiology",
            },
            "urgency": {
                "instructions": "What is the clinical urgency level?",
                "expected": "critical",
            },
        }

        output = self.engine.predict(situation, questions)

        self.assertIn("results", output)
        self.assertIn("metrics", output)
        results = output["results"]

        # Check constrained question
        self.assertIn("specialty", results)
        spec = results["specialty"]
        self.assertEqual(spec["type"], "CONSTRAINED")
        self.assertIn(spec["word"], ["cardiology", "neurology"])
        self.assertIn("probabilities", spec)
        self.assertIn("raw_mass", spec)
        self.assertIn("premise_ll", spec)
        self.assertIn("diag", spec)

        # Check open question
        self.assertIn("urgency", results)
        urg = results["urgency"]
        self.assertEqual(urg["type"], "OPEN")
        self.assertIsNotNone(urg["word"])
        self.assertIn("confidence", urg)
        self.assertIn("top_k", urg)
        self.assertIn("diag", urg)

    def test_precision_context_manager(self):
        self.assertEqual(self.engine.model_dtype(), torch.float32)
        # Cast to float64 temporarily
        with self.engine.precision(torch.float64):
            self.assertEqual(self.engine.model_dtype(), torch.float64)
        self.assertEqual(self.engine.model_dtype(), torch.float32)

    def test_singleton_engine_interface(self):
        eng = init_engine(
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu",
            dtype=torch.float32,
        )
        self.assertIs(get_engine(), eng)
        self.assertEqual(model_dtype(), torch.float32)

        out = predict_unified_tree(
            "State text",
            {"q": {"instructions": "What is this?", "candidates": ["a", "b"]}},
        )
        self.assertIn("q", out["results"])


if __name__ == "__main__":
    unittest.main()
