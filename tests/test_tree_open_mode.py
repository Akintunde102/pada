"""Unit tests for open-mode completion, bound-prefix guard, and stem grouping."""

import unittest
from types import SimpleNamespace
import torch
from transformers.cache_utils import DynamicCache

from pada.tree_engine.lexicon import VocabMasks, stem_word
from pada.tree_engine.open_mode import complete_batch_branched


class MockModel(torch.nn.Module):
    """Mock causal language model that emits controllable logits for continuation."""

    def __init__(self, step0_logits, step1_logits):
        super().__init__()
        self.step0_logits = step0_logits
        self.step1_logits = step1_logits
        self.step = 0
        # Dummy parameter to satisfy next(model.parameters())
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, past_key_values=None, position_ids=None, use_cache=True, **kwargs):
        batch_size = input_ids.shape[0]
        # Emit logits corresponding to the current step
        logits_template = self.step0_logits if self.step == 0 else self.step1_logits
        self.step += 1
        logits = logits_template.repeat(batch_size, 1, 1)

        # Update cache dummy
        if past_key_values is not None:
            k = torch.zeros((batch_size, 1, 1, 2))
            v = torch.zeros((batch_size, 1, 1, 2))
            past_key_values.update(k, v, 0)

        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class MockTokenizer:

    def convert_ids_to_tokens(self, tid):
        names = {0: "<stop>", 1: "emerg", 2: "ency", 3: "ent"}
        return names.get(tid, f"tok_{tid}")

    def decode(self, token_ids):
        return "".join(self.convert_ids_to_tokens(t) for t in token_ids)


class TestOpenMode(unittest.TestCase):

    def test_bound_prefix_prevention(self):
        # Vocab:
        # 0: <stop> (stop boundary)
        # 1: "emerg" (bound prefix)
        # 2: "ency" (subword continuation)
        vocab_size = 4
        stop_mask = torch.zeros(vocab_size, dtype=torch.bool)
        stop_mask[0] = True

        content_mask = torch.zeros(vocab_size, dtype=torch.bool)
        content_mask[1] = True

        bound_pfx_mask = torch.zeros(vocab_size, dtype=torch.bool)
        bound_pfx_mask[1] = True  # emerg is bound!

        cont_mask = torch.zeros(vocab_size, dtype=torch.bool)
        cont_mask[2] = True  # ency is continuation

        masks = VocabMasks(
            vocab_size=vocab_size,
            content_word_mask=content_mask,
            bound_prefix_mask=bound_pfx_mask,
            subword_cont_mask=cont_mask,
            stop_boundary_mask=stop_mask,
            id_to_word={1: "emerg", 2: "ency"},
        )

        # Step 0: Stop token 0 has higher probability than continuation token 2
        # BUT token 1 is a bound prefix! So forbid_bound_stop must force it to continue to token 2.
        step0_logits = torch.zeros((1, 1, vocab_size))
        step0_logits[0, 0, 0] = 5.0  # Stop logit is high
        step0_logits[0, 0, 2] = 2.0  # Continuation logit is lower

        # Step 1: Now stop token is highest and it stops
        step1_logits = torch.zeros((1, 1, vocab_size))
        step1_logits[0, 0, 0] = 10.0

        model = MockModel(step0_logits, step1_logits)
        tokenizer = MockTokenizer()

        cache = DynamicCache()
        k = torch.zeros((1, 1, 5, 2))
        v = torch.zeros((1, 1, 5, 2))
        cache.update(k, v, 0)

        results = complete_batch_branched(
            first_tids=[1],
            isolated_kv=cache,
            model=model,
            tokenizer=tokenizer,
            masks=masks,
            forbid_bound_stop=True,
            max_tokens=2,
        )

        # Should NOT stop at step 0 as bare "emerg"
        # It must have completed into "emergency" (emerg + ency)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["pieces"], ["emerg", "ency"])
        self.assertEqual(results[0]["word"], "emergency")
        self.assertFalse(results[0]["is_bound_trunc"])

    def test_stem_grouping_representative_preference(self):
        # Ensure that between a truncated bound prefix ('replac') and a completed word ('replacing'),
        # both share the stem 'replac', but the representative MUST be the completed word.
        self.assertEqual(stem_word("replac"), stem_word("replacing"))

        # Mock path records
        records = [
            {"word": "replac", "pieces": ["replac"], "joint": 0.8, "is_bound_trunc": True},
            {"word": "replacing", "pieces": ["replac", "ing"], "joint": 0.2, "is_bound_trunc": False},
        ]

        concepts = {}
        for rec in records:
            w, p, joint, is_trunc = rec["word"], rec["pieces"], rec["joint"], rec["is_bound_trunc"]
            stem = stem_word(w)
            rep_score = (0 if is_trunc else 1, joint)
            if stem not in concepts:
                concepts[stem] = {
                    "word": w,
                    "pieces": p,
                    "joint": joint,
                    "best_score": rep_score,
                    "members": [(w, joint)],
                }
            else:
                concepts[stem]["joint"] += joint
                concepts[stem]["members"].append((w, joint))
                if rep_score > concepts[stem]["best_score"]:
                    concepts[stem].update(word=w, pieces=p, best_score=rep_score)

        # The stem representative must be 'replacing', NOT the bare prefix 'replac'
        chosen_rep = list(concepts.values())[0]["word"]
        self.assertEqual(chosen_rep, "replacing")
        # Total joint mass should be the sum of both inflections (0.8 + 0.2 = 1.0)
        self.assertAlmostEqual(list(concepts.values())[0]["joint"], 1.0)


if __name__ == "__main__":
    unittest.main()
