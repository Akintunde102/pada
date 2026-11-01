"""Contract limits and constraints for Jev.

These limits are frozen upfront to guarantee bounded latency and memory usage
in both prefill and branch batch execution.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractLimits:
    # Maximum characters in state text (~32,768 tokens worst-case)
    MAX_STATE_CHARS: int = 131_072
    
    # Maximum number of concurrent question branches per request
    MAX_QUESTIONS_PER_REQUEST: int = 256
    
    # Maximum number of options per choice question
    MAX_OPTIONS_PER_QUESTION: int = 256
    
    # Minimum options for a choice question
    MIN_OPTIONS_PER_QUESTION: int = 2
    
    # Maximum instructions length per question in characters
    MAX_INSTRUCTION_CHARS: int = 4_096
    
    # Maximum criteria description length per option in characters
    MAX_CRITERIA_CHARS: int = 2_048
    
    # Score questions constraints
    MIN_SCORE_LEVELS: int = 2
    MAX_SCORE_LEVELS: int = 100
    DEFAULT_SCORE_LEVELS: int = 5


DEFAULT_LIMITS = ContractLimits()
