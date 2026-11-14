"""Lexicon, stemming, and tokenization boundary utilities."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import torch

from .constants import (
    IRREGULAR_STEM_EXCEPTIONS,
    KNOWN_BOUND_PREFIXES,
    GRAMMAR_GLUE,
    QUESTION_STOPWORDS,
)

# Initialize stemmer: Snowball with rule-based fallback
_snowball = None
try:
    from nltk.stem import SnowballStemmer
    _snowball = SnowballStemmer("english")
except Exception:
    _snowball = None


def stem_word(w: str) -> str:
    """Stem a word using Snowball if available, or a deterministic fallback."""
    if _snowball is not None:
        try:
            return _snowball.stem(w.lower().strip())
        except Exception:
            pass

    word = w.lower().strip()
    if word in IRREGULAR_STEM_EXCEPTIONS:
        return word
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-1] if word[:-1].endswith("e") else word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


_english_lexicon: Optional[Set[str]] = None
_lexicon_attempted: bool = False


def get_english_lexicon() -> Optional[Set[str]]:
    """Load or retrieve the English lexicon cache if available."""
    global _english_lexicon, _lexicon_attempted
    if _english_lexicon is not None or _lexicon_attempted:
        return _english_lexicon

    _lexicon_attempted = True
    try:
        import nltk
        from nltk.corpus import words as _nltk_words
        try:
            _english_lexicon = set(w.lower() for w in _nltk_words.words())
        except LookupError:
            nltk.download("words", quiet=True)
            _english_lexicon = set(w.lower() for w in _nltk_words.words())
    except Exception:
        _english_lexicon = None
    return _english_lexicon


def is_bound_bpe_prefix(clean: str, all_space_words: Set[str], lexicon: Optional[Set[str]] = None) -> bool:
    """True if `clean` is a truncated BPE subword root (like 'c' in 'cello', 'emerg', or 'indemn') rather than a complete word."""
    if len(clean) == 1:
        return True
    if clean in KNOWN_BOUND_PREFIXES:
        return True
    active_lexicon = lexicon if lexicon is not None else get_english_lexicon()
    if active_lexicon is not None and clean not in active_lexicon:
        # If it is not in the dictionary and longer words in the vocab start with it, it is a bound prefix
        if any(w.startswith(clean) and len(w) >= len(clean) + 2 for w in all_space_words):
            return True
    return False


@dataclass
class VocabMasks:
    """Vocabulary masks and lookup dictionaries for guided generation."""
    vocab_size: int
    content_word_mask: torch.Tensor
    bound_prefix_mask: torch.Tensor
    subword_cont_mask: torch.Tensor
    stop_boundary_mask: torch.Tensor
    id_to_word: Dict[int, str]


def build_vocab_masks(tokenizer, vocab_size: int, device: str = "cpu") -> VocabMasks:
    """Construct token classification masks for vocabulary tokens."""
    content_word_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    bound_prefix_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    subword_cont_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    stop_boundary_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    id_to_word: Dict[int, str] = {}

    tok_ids = list(range(len(tokenizer)))
    decoded = tokenizer.batch_decode([[i] for i in tok_ids], clean_up_tokenization_spaces=False)

    all_space_words = set()
    for text in decoded:
        if text and text.startswith(" "):
            c = text.strip().lower()
            if c.isalpha() and len(c) >= 2:
                all_space_words.add(c)

    lexicon = get_english_lexicon()

    for tid, text in enumerate(decoded):
        if not text:
            continue
        if text[0] in (" ", "\n", ".", ",", ";", ":", "!", "?"):
            stop_boundary_mask[tid] = True
        if text.startswith(" "):
            clean = text.strip().lower()
            if clean.isalpha() and len(clean) >= 1 and clean not in GRAMMAR_GLUE:
                content_word_mask[tid] = True
                id_to_word[tid] = clean
                if is_bound_bpe_prefix(clean, all_space_words, lexicon=lexicon):
                    bound_prefix_mask[tid] = True
        else:
            if text.isalpha() and text.islower() and text.isascii():
                subword_cont_mask[tid] = True
                id_to_word[tid] = text

    vocab = tokenizer.get_vocab()
    for eos_tok in ["<|im_end|>", "<|endoftext|>"]:
        if eos_tok in vocab and vocab[eos_tok] < vocab_size:
            stop_boundary_mask[vocab[eos_tok]] = True

    return VocabMasks(
        vocab_size=vocab_size,
        content_word_mask=content_word_mask,
        bound_prefix_mask=bound_prefix_mask,
        subword_cont_mask=subword_cont_mask,
        stop_boundary_mask=stop_boundary_mask,
        id_to_word=id_to_word,
    )


def find_question_premise_offsets(branch_ids: List[int], id_to_word: Dict[int, str], tokenizer) -> List[int]:
    """
    Identify the token offsets inside `Question: <instr> One-word answer:` that correspond to
    domain-specific premise words in `<instr>` (e.g. 'cardiovascular', 'pulmonary', 'avian').
    """
    suffix_len = len(tokenizer.encode(" One-word answer:", add_special_tokens=False))
    prefix_len = len(tokenizer.encode("Question:", add_special_tokens=False))
    body_end = max(prefix_len + 1, len(branch_ids) - suffix_len)

    premise_offsets = []
    for offset in range(prefix_len, body_end):
        tid = branch_ids[offset]
        w = id_to_word.get(tid, tokenizer.decode([tid]).strip().lower())
        if w and w.isalpha() and len(w) >= 3 and w not in QUESTION_STOPWORDS:
            premise_offsets.append(offset)
    if not premise_offsets:
        premise_offsets = list(range(prefix_len, body_end))
    return premise_offsets
