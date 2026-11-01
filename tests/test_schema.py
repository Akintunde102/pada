"""Comprehensive Unit Tests for Jev Phase 0: Schema, Contracts, and Key Stripping."""

import unittest
from pydantic import ValidationError

from pada.schema.billing import calculate_output_tokens
from pada.schema.internal import HostModelAdapter, StrippedModelInput
from pada.schema.limits import DEFAULT_LIMITS
from pada.schema.request import (
    ChoiceQuestion,
    JevRequest,
    NoulQuestion,
    ScoreQuestion,
)
from pada.schema.response import (
    ChoiceResult,
    JevResponse,
    NoulResult,
    ScoreResult,
    Usage,
)


class TestJevSchema(unittest.TestCase):
    def setUp(self):
        # The canonical customer support example from Hume's paper
        self.sample_request_data = {
            "state": "My payouts have failed three times. The bank says everything is fine. Can someone please fix this?",
            "questions": {
                "queue": {
                    "type": "choice",
                    "instructions": "Which team should handle this ticket?",
                    "criteria": {
                        "payments": "Payout failures and payment processing",
                        "account": "Login and account access",
                        "other": "Something else",
                    },
                },
                "escalate": {
                    "type": "noul",
                    "instructions": "Does this message require urgent human attention?",
                },
            },
        }

    def test_canonical_request_parsing(self):
        req = JevRequest.model_validate(self.sample_request_data)
        self.assertIn("queue", req.questions)
        self.assertIn("escalate", req.questions)
        self.assertIsInstance(req.questions["queue"], ChoiceQuestion)
        self.assertIsInstance(req.questions["escalate"], NoulQuestion)
        self.assertEqual(len(req.questions["queue"].criteria), 3)

    def test_score_question_parsing(self):
        score_data = {
            "state": "Customer was very rude and demanded an immediate refund.",
            "questions": {
                "sentiment": {
                    "type": "score",
                    "instructions": "Rate customer anger from 1 (calm) to 5 (furious)",
                    "levels": 5,
                    "criteria": {
                        0: "Very calm",
                        4: "Extremely angry",
                    },
                }
            },
        }
        req = JevRequest.model_validate(score_data)
        q = req.questions["sentiment"]
        self.assertIsInstance(q, ScoreQuestion)
        self.assertEqual(q.levels, 5)

    def test_forbidden_extra_fields(self):
        data = dict(self.sample_request_data)
        data["unauthorized_field"] = "bad"
        with self.assertRaises(ValidationError):
            JevRequest.model_validate(data)

    def test_validation_empty_state(self):
        with self.assertRaises(ValidationError):
            JevRequest(state="", questions=self.sample_request_data["questions"])

    def test_validation_zero_questions(self):
        with self.assertRaises(ValidationError):
            JevRequest(state="Some state", questions={})

    def test_choice_criteria_limits(self):
        # Less than 2 options
        with self.assertRaises(ValidationError):
            ChoiceQuestion(
                instructions="Pick one",
                criteria={"only_one": "Single option description"},
            )

        # Empty option key
        with self.assertRaises(ValidationError):
            ChoiceQuestion(
                instructions="Pick one",
                criteria={"": "Empty key", "opt2": "Desc"},
            )

        # Empty description
        with self.assertRaises(ValidationError):
            ChoiceQuestion(
                instructions="Pick one",
                criteria={"opt1": "   ", "opt2": "Valid"},
            )

    def test_score_criteria_bounds(self):
        with self.assertRaises(ValidationError):
            ScoreQuestion(
                instructions="Rate quality",
                levels=5,
                criteria={5: "Out of bounds level (valid: 0 to 4)"},
            )

    def test_valid_response_construction(self):
        res = JevResponse(
            results={
                "queue": ChoiceResult(
                    probabilities={"payments": 0.91, "account": 0.06, "other": 0.03}
                ),
                "escalate": NoulResult(p_true=0.42),
            },
            usage=Usage(input_tokens=45, output_tokens=78),
        )
        self.assertEqual(res["escalate"].p_true, 0.42)
        self.assertAlmostEqual(res["queue"].probabilities["payments"], 0.91)
        self.assertEqual(res.usage.input_tokens, 45)

    def test_invalid_choice_probabilities_sum(self):
        with self.assertRaises(ValidationError):
            ChoiceResult(probabilities={"opt1": 0.5, "opt2": 0.1})  # Sum = 0.6 != 1.0

    def test_hard_rule_key_stripping_and_renaming_invariance(self):
        """Crucial test of Hume's finding:
        Altering question keys or option keys must result in IDENTICAL model input.
        The model never sees the keys.
        """
        req_original = JevRequest.model_validate(self.sample_request_data)

        # Create a modified request with completely renamed question and option keys,
        # but IDENTICAL instructions and criteria text:
        renamed_request_data = {
            "state": "My payouts have failed three times. The bank says everything is fine. Can someone please fix this?",
            "questions": {
                "RENAMED_QUESTION_ALPHA": {
                    "type": "choice",
                    "instructions": "Which team should handle this ticket?",
                    "criteria": {
                        "KEY_X": "Payout failures and payment processing",
                        "KEY_Y": "Login and account access",
                        "KEY_Z": "Something else",
                    },
                },
                "RENAMED_QUESTION_BETA": {
                    "type": "noul",
                    "instructions": "Does this message require urgent human attention?",
                },
            },
        }
        req_renamed = JevRequest.model_validate(renamed_request_data)

        stripped_orig, map_orig = HostModelAdapter.strip_request(req_original)
        stripped_renamed, map_renamed = HostModelAdapter.strip_request(req_renamed)

        # 1. The stripped model inputs must be structurally and textually EQUAL
        self.assertEqual(stripped_orig.state, stripped_renamed.state)
        self.assertEqual(len(stripped_orig.branches), len(stripped_renamed.branches))

        for b_orig, b_renamed in zip(stripped_orig.branches, stripped_renamed.branches):
            self.assertEqual(b_orig.index, b_renamed.index)
            self.assertEqual(b_orig.question_type, b_renamed.question_type)
            self.assertEqual(b_orig.instructions, b_renamed.instructions)
            self.assertEqual(len(b_orig.options), len(b_renamed.options))
            for opt_o, opt_r in zip(b_orig.options, b_renamed.options):
                self.assertEqual(opt_o.index, opt_r.index)
                self.assertEqual(opt_o.text, opt_r.text)

        # 2. Check that no question keys or option keys exist in stripped input
        for key in ["queue", "escalate", "payments", "account", "other"]:
            # Confirm keys are not stored in any attribute of stripped_orig
            for b in stripped_orig.branches:
                self.assertNotEqual(b.instructions, key)
                for opt in b.options:
                    self.assertNotEqual(opt.text, key)

    def test_response_reconstruction(self):
        req = JevRequest.model_validate(self.sample_request_data)
        stripped_input, mapping = HostModelAdapter.strip_request(req)

        # Simulated raw model outputs from neural heads
        # branch 0 (choice with 3 options): [0.91, 0.06, 0.03]
        # branch 1 (noul): 0.42
        raw_outputs = {
            0: [0.91, 0.06, 0.03],
            1: 0.42,
        }
        usage = Usage(input_tokens=52, output_tokens=79)

        response = HostModelAdapter.reconstruct_response(mapping, raw_outputs, usage)

        self.assertIn("queue", response.results)
        self.assertIn("escalate", response.results)
        queue_res = response.results["queue"]
        self.assertIsInstance(queue_res, ChoiceResult)
        self.assertEqual(queue_res.probabilities["payments"], 0.91)
        self.assertEqual(queue_res.probabilities["account"], 0.06)
        self.assertEqual(queue_res.probabilities["other"], 0.03)

        escalate_res = response.results["escalate"]
        self.assertIsInstance(escalate_res, NoulResult)
        self.assertEqual(escalate_res.p_true, 0.42)

    def test_synthetic_billing_formula(self):
        req = JevRequest.model_validate(self.sample_request_data)
        tokens_base = calculate_output_tokens(req.questions)
        
        # Formula:
        # Base shared = 4
        # 'queue' key tokens + (3 options * 15 = 45)
        # 'escalate' key tokens + (1 answer * 15 = 15)
        self.assertGreater(tokens_base, 4 + 45 + 15)

        # Hume's probe finding: changing key length changes output_tokens
        modified_data = dict(self.sample_request_data)
        modified_data["questions"] = dict(self.sample_request_data["questions"])
        modified_data["questions"]["queue_with_very_long_custom_identifier_name_for_testing"] = (
            modified_data["questions"].pop("queue")
        )
        req_longer_key = JevRequest.model_validate(modified_data)
        tokens_longer = calculate_output_tokens(req_longer_key.questions)

        self.assertGreater(tokens_longer, tokens_base)


if __name__ == "__main__":
    unittest.main()
