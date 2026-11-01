"""Host-to-Model Internal Representation & Adapter.

ENFORCES THE HARD ARCHITECTURAL RULE:
"Question keys and option keys are never sent to the model.
They exist only on the host side, for mapping results back.
The model sees only the instructions and criteria text."
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from pada.schema.request import ChoiceQuestion, JevRequest, NoulQuestion, ScoreQuestion
from pada.schema.response import (
    ChoiceResult,
    JevResponse,
    NoulResult,
    QuestionResult,
    ScoreResult,
    Usage,
)


@dataclass(frozen=True)
class AnonymousOption:
    """An option stripped of any user-facing identifier key."""
    index: int
    text: str


@dataclass(frozen=True)
class AnonymousQuestion:
    """A question stripped of its user-facing key.
    
    Contains strictly the instructions, question type, and criteria texts.
    Options preserve their original declaration order for in-context attention.
    """
    index: int
    question_type: Literal["noul", "choice", "score"]
    instructions: str
    options: List[AnonymousOption] = field(default_factory=list)
    score_levels: Optional[int] = None
    score_criteria: Optional[Dict[int, str]] = None


@dataclass(frozen=True)
class StrippedModelInput:
    """The clean data structure passed down to the neural backbone / token serializer.
    
    Contains ZERO user keys.
    """
    state: str
    branches: List[AnonymousQuestion]


@dataclass
class BranchKeyMapping:
    """Host-side bookkeeping to map branch index back to user question key and options."""
    question_key: str
    question_type: Literal["noul", "choice", "score"]
    # Maps internal option index [0, 1, ..., N-1] to user option key
    option_index_to_key: Dict[int, str] = field(default_factory=dict)
    # Number of levels for score questions
    score_levels: Optional[int] = None


@dataclass
class RequestKeyMapping:
    """Complete host-side mapping table for a request."""
    # Maps branch index [0, 1, ..., Q-1] to branch mapping
    branches: Dict[int, BranchKeyMapping] = field(default_factory=dict)


class HostModelAdapter:
    """Enforces the boundary between the host API layer and the internal neural model."""

    @staticmethod
    def strip_request(request: JevRequest) -> Tuple[StrippedModelInput, RequestKeyMapping]:
        """Strips all user question keys and option keys from the request.
        
        Returns:
            StrippedModelInput: Input suitable for model serialization, devoid of user keys.
            RequestKeyMapping: Host-side table required to reconstruct user keys upon completion.
        """
        stripped_branches: List[AnonymousQuestion] = []
        mapping = RequestKeyMapping()

        for branch_idx, (q_key, question) in enumerate(request.questions.items()):
            if isinstance(question, NoulQuestion):
                stripped_branches.append(
                    AnonymousQuestion(
                        index=branch_idx,
                        question_type="noul",
                        instructions=question.instructions,
                    )
                )
                mapping.branches[branch_idx] = BranchKeyMapping(
                    question_key=q_key,
                    question_type="noul",
                )

            elif isinstance(question, ChoiceQuestion):
                anon_options: List[AnonymousOption] = []
                opt_map: Dict[int, str] = {}

                # Preserve dictionary iteration order (Python 3.7+ guarantees insertion order)
                for opt_idx, (opt_key, opt_desc) in enumerate(question.criteria.items()):
                    anon_options.append(
                        AnonymousOption(index=opt_idx, text=opt_desc)
                    )
                    opt_map[opt_idx] = opt_key

                stripped_branches.append(
                    AnonymousQuestion(
                        index=branch_idx,
                        question_type="choice",
                        instructions=question.instructions,
                        options=anon_options,
                    )
                )
                mapping.branches[branch_idx] = BranchKeyMapping(
                    question_key=q_key,
                    question_type="choice",
                    option_index_to_key=opt_map,
                )

            elif isinstance(question, ScoreQuestion):
                stripped_branches.append(
                    AnonymousQuestion(
                        index=branch_idx,
                        question_type="score",
                        instructions=question.instructions,
                        score_levels=question.levels,
                        score_criteria=question.criteria,
                    )
                )
                mapping.branches[branch_idx] = BranchKeyMapping(
                    question_key=q_key,
                    question_type="score",
                    score_levels=question.levels,
                )
            else:
                raise TypeError(f"Unknown question type: {type(question)}")

        model_input = StrippedModelInput(
            state=request.state,
            branches=stripped_branches,
        )
        return model_input, mapping

    @staticmethod
    def reconstruct_response(
        mapping: RequestKeyMapping,
        raw_outputs: Dict[int, Any],
        usage: Usage,
    ) -> JevResponse:
        """Takes numerical outputs from the readout heads and maps them back to user keys.
        
        Args:
            mapping: Host-side mapping table generated by strip_request.
            raw_outputs: Dict mapping branch index -> raw head output:
                         - for noul: float p_true in [0, 1]
                         - for choice: list of float probabilities [p0, p1, ..., pN-1]
                         - for score: tuple(distribution: list[float], expected: float)
            usage: Token accounting usage.
            
        Returns:
            JevResponse: Fully populated user response with validated keys and probabilities.
        """
        results: Dict[str, QuestionResult] = {}

        for branch_idx, branch_map in mapping.branches.items():
            if branch_idx not in raw_outputs:
                raise KeyError(f"Missing model head output for branch index {branch_idx}.")

            raw_val = raw_outputs[branch_idx]
            q_key = branch_map.question_key

            if branch_map.question_type == "noul":
                p_true = float(raw_val)
                results[q_key] = NoulResult(p_true=p_true)

            elif branch_map.question_type == "choice":
                probs_list = list(raw_val)
                if len(probs_list) != len(branch_map.option_index_to_key):
                    raise ValueError(
                        f"Option count mismatch for branch {branch_idx}: "
                        f"expected {len(branch_map.option_index_to_key)}, got {len(probs_list)}."
                    )
                choice_probs: Dict[str, float] = {}
                for opt_idx, opt_key in branch_map.option_index_to_key.items():
                    choice_probs[opt_key] = float(probs_list[opt_idx])
                results[q_key] = ChoiceResult(probabilities=choice_probs)

            elif branch_map.question_type == "score":
                if isinstance(raw_val, tuple):
                    dist, expected = raw_val
                elif isinstance(raw_val, dict):
                    dist = raw_val["distribution"]
                    expected = raw_val["expected"]
                else:
                    raise TypeError(f"Unsupported score head output format: {type(raw_val)}")
                results[q_key] = ScoreResult(
                    distribution=[float(x) for x in dist],
                    expected=float(expected),
                )

        return JevResponse(results=results, usage=usage)
