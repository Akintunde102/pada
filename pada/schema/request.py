"""User-facing Request Schema for Jev.

Complies with the Jev public API specification:
- state: context string
- questions: mapping of arbitrary question_key -> Question specification
"""

from typing import Annotated, Dict, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pada.schema.limits import DEFAULT_LIMITS


class BaseQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: str = Field(
        ...,
        min_length=1,
        max_length=DEFAULT_LIMITS.MAX_INSTRUCTION_CHARS,
        description="The instruction prompt for this question.",
    )


class NoulQuestion(BaseQuestion):
    """Binary decision question (TypeSafe's Yes/No type)."""
    type: Literal["noul"] = "noul"


class ChoiceQuestion(BaseQuestion):
    """Multi-choice categorical question selecting among mutually exclusive options."""
    type: Literal["choice"] = "choice"
    criteria: Dict[str, str] = Field(
        ...,
        description="Dictionary mapping user option keys to their descriptive criteria.",
    )

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, v: Dict[str, str]) -> Dict[str, str]:
        if len(v) < DEFAULT_LIMITS.MIN_OPTIONS_PER_QUESTION:
            raise ValueError(
                f"Choice question must have at least {DEFAULT_LIMITS.MIN_OPTIONS_PER_QUESTION} options, got {len(v)}."
            )
        if len(v) > DEFAULT_LIMITS.MAX_OPTIONS_PER_QUESTION:
            raise ValueError(
                f"Choice question exceeds maximum of {DEFAULT_LIMITS.MAX_OPTIONS_PER_QUESTION} options, got {len(v)}."
            )
        for opt_key, desc in v.items():
            if not opt_key.strip():
                raise ValueError("Option key cannot be empty or whitespace.")
            if not desc.strip():
                raise ValueError(f"Criteria description for option '{opt_key}' cannot be empty.")
            if len(desc) > DEFAULT_LIMITS.MAX_CRITERIA_CHARS:
                raise ValueError(
                    f"Criteria description for option '{opt_key}' exceeds max length of {DEFAULT_LIMITS.MAX_CRITERIA_CHARS} chars."
                )
        return v


class ScoreQuestion(BaseQuestion):
    """Ordinal scoring question predicting probability distribution across discrete ordered levels."""
    type: Literal["score"] = "score"
    levels: int = Field(
        default=DEFAULT_LIMITS.DEFAULT_SCORE_LEVELS,
        ge=DEFAULT_LIMITS.MIN_SCORE_LEVELS,
        le=DEFAULT_LIMITS.MAX_SCORE_LEVELS,
        description="Number of discrete rating levels (e.g. 5 for a 1-5 scale).",
    )
    criteria: Optional[Dict[int, str]] = Field(
        default=None,
        description="Optional mapping of integer levels to qualitative descriptions.",
    )

    @model_validator(mode="after")
    def validate_score_criteria(self) -> "ScoreQuestion":
        if self.criteria is not None:
            for level, desc in self.criteria.items():
                if level < 0 or level >= self.levels:
                    raise ValueError(
                        f"Criteria level index {level} out of range for levels count {self.levels} (0 to {self.levels - 1})."
                    )
                if not desc.strip():
                    raise ValueError(f"Description for score level {level} cannot be empty.")
        return self


# Discriminated union of question types
Question = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion],
    Field(discriminator="type"),
]


class JevRequest(BaseModel):
    """Top-level Jev inference request."""
    model_config = ConfigDict(extra="forbid")

    state: str = Field(
        ...,
        min_length=1,
        max_length=DEFAULT_LIMITS.MAX_STATE_CHARS,
        description="Shared state / context processed once across all questions.",
    )
    questions: Dict[str, Question] = Field(
        ...,
        description="Mapping from user question key to question definition.",
    )

    @field_validator("questions")
    @classmethod
    def validate_questions_map(cls, v: Dict[str, Question]) -> Dict[str, Question]:
        if len(v) == 0:
            raise ValueError("Request must contain at least 1 question.")
        if len(v) > DEFAULT_LIMITS.MAX_QUESTIONS_PER_REQUEST:
            raise ValueError(
                f"Request exceeds maximum of {DEFAULT_LIMITS.MAX_QUESTIONS_PER_REQUEST} questions, got {len(v)}."
            )
        for key in v.keys():
            if not key.strip():
                raise ValueError("Question key cannot be empty or whitespace.")
        return v
