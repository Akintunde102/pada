"""Unit tests for lexicon, stemming, and vocabulary masks."""

import unittest
import torch

from pada.tree_engine.constants import (
    IRREGULAR_STEM_EXCEPTIONS,
    KNOWN_BOUND_PREFIXES,
)
from pada.tree_engine.lexicon import (
    stem_word,
    is_bound_bpe_prefix,
    find_question_premise_offsets,
    build_vocab_masks,
)


class MockTokenizer:
    """Minimal mock tokenizer for token-level testing."""

    def __init__(self, vocab_dict=None):
        if vocab_dict is None:
            vocab_dict = {
                "<|endoftext|>": 0,
                "<|im_end|>": 1,
                " ": 2,
                " the": 3,
                " emergency": 4,
                " emerg": 5,
                " hospital": 6,
                " hosp": 7,
                "ency": 8,
                "ital": 9,
                "Question:": 10,
                " What": 11,
                " cardiovascular": 12,
                " One-word": 13,
                " answer:": 14,
            }
        self.vocab = vocab_dict
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        # Deterministic mock encoding mapping known substrings
        if text == "Question:":
            return [10]
        if text == " One-word answer:":
            return [13, 14]
        tokens = []
        for word, tid in sorted(self.vocab.items(), key=lambda kv: -len(kv[0])):
            if word.strip() and word in text:
                tokens.append(tid)
        return tokens if tokens else [2]

    def decode(self, token_ids):
        return "".join(self.inv_vocab.get(t, "") for t in token_ids)

    def batch_decode(self, sequences, clean_up_tokenization_spaces=False):
        return [self.decode(seq) for seq in sequences]

    def convert_ids_to_tokens(self, tid):
        return self.inv_vocab.get(tid, f"<unk_{tid}>")


class TestLexicon(unittest.TestCase):

    def test_stem_word_exceptions(self):
        for w in IRREGULAR_STEM_EXCEPTIONS:
            self.assertEqual(stem_word(w), w)

    def test_stem_word_rules(self):
        # Test basic inflectional reductions
        self.assertEqual(stem_word("running"), "runn")
        self.assertEqual(stem_word("boxes"), "box")
        self.assertEqual(stem_word("cats"), "cat")
        # Word ending in ed
        self.assertTrue(stem_word("jumped") in ("jump", "jumpe"))

    def test_is_bound_bpe_prefix(self):
        # Direct check against known bound prefixes
        self.assertTrue(is_bound_bpe_prefix("emerg", {"emergency"}))
        self.assertTrue(is_bound_bpe_prefix("indemn", {"indemnification"}))
        self.assertTrue(is_bound_bpe_prefix("card", {"cardiology"}))

        # Words in lexicon or full standalone words shouldn't trigger if not in known prefixes
        lexicon = {"hospital", "cat", "emergency"}
        self.assertFalse(is_bound_bpe_prefix("hospital", {"hospital", "hospitals"}, lexicon=lexicon))
        self.assertFalse(is_bound_bpe_prefix("cat", {"cat", "cats"}, lexicon=lexicon))

    def test_build_vocab_masks(self):
        tok = MockTokenizer()
        masks = build_vocab_masks(tok, vocab_size=len(tok), device="cpu")

        self.assertEqual(masks.vocab_size, len(tok))
        # EOS tokens should have stop_boundary_mask True
        self.assertTrue(masks.stop_boundary_mask[0].item())
        self.assertTrue(masks.stop_boundary_mask[1].item())

        # " emergency" is a space word
        emerg_id = tok.vocab[" emergency"]
        self.assertTrue(masks.content_word_mask[emerg_id].item())
        self.assertEqual(masks.id_to_word[emerg_id], "emergency")

        # " emerg" is a bound prefix
        emerg_prefix_id = tok.vocab[" emerg"]
        self.assertTrue(masks.bound_prefix_mask[emerg_prefix_id].item())

        # "ency" without leading space is subword continuation
        ency_id = tok.vocab["ency"]
        self.assertTrue(masks.subword_cont_mask[ency_id].item())

    def test_find_question_premise_offsets(self):
        tok = MockTokenizer()
        branch_ids = [10, 11, 12, 13, 14]
        id_to_word = {12: "cardiovascular"}
        offsets = find_question_premise_offsets(branch_ids, id_to_word, tok)

        # Offset for index 2 ("cardiovascular") should be included, while Question and One-word answer are trimmed
        self.assertIn(2, offsets)


if __name__ == "__main__":
    unittest.main()
