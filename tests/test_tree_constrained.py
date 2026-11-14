"""Unit tests for constrained scoring, probability normalization, and OOD triggering."""

import unittest
import math
from pada.tree_engine.constrained import (
    encode_candidate,
    finalize_constrained_dict,
)


class MockTokenizer:

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class TestConstrained(unittest.TestCase):

    def test_encode_candidate(self):
        tok = MockTokenizer()
        encoded = encode_candidate("apple_pie", tok)
        # Should start with space and replace underscore with space
        decoded_chars = [chr(t) for t in encoded]
        self.assertEqual("".join(decoded_chars), " apple pie")

    def test_finalize_constrained_dict(self):
        candidates = ["apple", "banana", "cherry"]
        # log joints: e.g. -1.0, -2.0, -3.0
        log_joints = {"apple": -1.0, "banana": -2.0, "cherry": -3.0}
        details = {c: {"log_joint": log_joints[c]} for c in candidates}

        res = finalize_constrained_dict(
            candidates=candidates,
            log_joints=log_joints,
            details=details,
            premise_ll=-2.0,
            ood_mass_threshold=0.01,
            ood_premise_threshold=-6.0,
        )

        self.assertEqual(res["word"], "apple")
        self.assertEqual(res["runner_up"][0], "banana")
        # Sum of probabilities should be ~ 1.0
        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)
        self.assertFalse(res["is_ood"])

    def test_ood_triggered_by_raw_mass(self):
        candidates = ["apple", "banana"]
        log_joints = {"apple": -10.0, "banana": -11.0}
        details = {c: {"log_joint": log_joints[c]} for c in candidates}

        res = finalize_constrained_dict(
            candidates=candidates,
            log_joints=log_joints,
            details=details,
            premise_ll=-2.0,
            ood_mass_threshold=0.01,
            ood_premise_threshold=-6.0,
        )

        self.assertTrue(res["is_ood"])
        self.assertLess(res["raw_mass"], 0.01)

    def test_ood_triggered_by_premise_ll(self):
        candidates = ["apple", "banana"]
        log_joints = {"apple": -1.0, "banana": -2.0}
        details = {c: {"log_joint": log_joints[c]} for c in candidates}

        res = finalize_constrained_dict(
            candidates=candidates,
            log_joints=log_joints,
            details=details,
            premise_ll=-8.5,  # lower than -6.0
            ood_mass_threshold=0.01,
            ood_premise_threshold=-6.0,
        )

        self.assertTrue(res["is_ood"])


if __name__ == "__main__":
    unittest.main()
