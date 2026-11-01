"""Synthetic Token Billing Formula for Jev.

As discovered by Archer Hume during the 10,000-call empirical probe:
'output_tokens' returned by TypeSafe's Jev is NOT generated text tokens.
It is a synthetic, deterministic host-side billing formula.
"""

from typing import Dict, Protocol
from pada.schema.request import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion


class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


class DefaultCharTokenizer:
    """Fallback rough tokenizer when a neural tokenizer is not specified (approx ~4 chars/token)."""
    def encode(self, text: str) -> list[int]:
        # Approximate 1 token per 4 characters, minimum 1 token for non-empty text
        if not text:
            return []
        token_count = max(1, (len(text) + 3) // 4)
        return list(range(token_count))


def calculate_output_tokens(
    questions: Dict[str, Question],
    tokenizer: TokenizerProtocol | None = None,
) -> int:
    """Calculates the synthetic output_tokens billed by Jev.

    Formula verified by Hume:
    - Base shared overhead: 4 tokens
    - Per question answer slot: 15 tokens per allowed discrete answer
      * noul has 1 binary answer slot (15 tokens)
      * choice has N option slots (15 * N tokens)
      * score has K ordinal level slots (15 * K tokens)
    - Host-side question key length: len(tokenizer.encode(question_key))
      (This proved output_tokens was synthetic, because question_key is never sent to the model!)
    """
    tok = tokenizer if tokenizer is not None else DefaultCharTokenizer()
    
    # Hume observed 4 shared base tokens
    total_tokens = 4

    for q_key, q in questions.items():
        # Length of the host key in tokens
        key_tokens = len(tok.encode(q_key))
        total_tokens += key_tokens

        if isinstance(q, NoulQuestion):
            # noul: 1 binary slot = 15 tokens
            total_tokens += 15
        elif isinstance(q, ChoiceQuestion):
            # choice: 15 tokens per option
            num_options = len(q.criteria)
            total_tokens += 15 * num_options
        elif isinstance(q, ScoreQuestion):
            # score: 15 tokens per level
            total_tokens += 15 * q.levels
        else:
            # Fallback
            total_tokens += 15

    return total_tokens
