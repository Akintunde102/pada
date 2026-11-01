"""User-facing Response Schema for Jev.

Returns calibrated probability readouts and synthetic token billing usage.
"""

from typing import Dict, List, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_tokens: int = Field(ge=0, description="Total input tokens (state + question branches).")
    output_tokens: int = Field(ge=0, description="Synthetic post-inference billed tokens formula.")


class NoulResult(BaseModel):
    """Result for a binary noul question."""
    model_config = ConfigDict(extra="forbid")
    p_true: float = Field(..., ge=0.0, le=1.0, description="Calibrated posterior probability P(True).")


class ChoiceResult(BaseModel):
    """Result for a multi-choice question."""
    model_config = ConfigDict(extra="forbid")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution mapped back to original user option keys.",
    )

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: Dict[str, float]) -> Dict[str, float]:
        for k, p in v.items():
            if p < 0.0 or p > 1.0:
                raise ValueError(f"Probability for '{k}' must be between 0.0 and 1.0, got {p}.")
        total = sum(v.values())
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Choice probabilities must sum to approximately 1.0, got {total}.")
        return v


class ScoreResult(BaseModel):
    """Result for an ordinal scoring question."""
    model_config = ConfigDict(extra="forbid")
    distribution: List[float] = Field(
        ...,
        description="Probability distribution over ordinal levels [0, ..., K-1].",
    )
    expected: float = Field(
        ...,
        description="Probability-weighted expectation E[S] = sum(k * P(S=k)).",
    )

    @field_validator("distribution")
    @classmethod
    def validate_distribution(cls, v: List[float]) -> List[float]:
        for idx, p in enumerate(v):
            if p < 0.0 or p > 1.0:
                raise ValueError(f"Probability at level {idx} must be between 0.0 and 1.0, got {p}.")
        total = sum(v)
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Score distribution probabilities must sum to approximately 1.0, got {total}.")
        return v


QuestionResult = Union[NoulResult, ChoiceResult, ScoreResult]


class JevResponse(BaseModel):
    """Top-level Jev inference response."""
    model_config = ConfigDict(extra="forbid")

    results: Dict[str, QuestionResult] = Field(
        ...,
        description="Mapping from user question key to calibrated decision result.",
    )
    usage: Usage = Field(
        ...,
        description="Token accounting and synthetic billing metrics.",
    )

    def __getitem__(self, key: str) -> QuestionResult:
        """Allow dict-like indexing response[question_key]."""
        return self.results[key]

    def __contains__(self, key: str) -> bool:
        return key in self.results
