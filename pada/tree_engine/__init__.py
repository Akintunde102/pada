"""Unified Tree Attention Engine for Parallel Multi-Question LLM Inference."""

from .constants import (
    P_STOP_FLOOR,
    DEFAULT_OOD_MASS_THRESHOLD,
    DEFAULT_OOD_PREMISE_THRESHOLD,
    SUBWORD_BRANCH_FACTOR,
    SUBWORD_BRANCH_RATIO,
    DIAG_MODE,
    KNOWN_BOUND_PREFIXES,
    IRREGULAR_STEM_EXCEPTIONS,
    GRAMMAR_GLUE,
    QUESTION_STOPWORDS,
    GUIDED_ANCHOR_PROMPT,
)
from .lexicon import (
    stem_word,
    is_bound_bpe_prefix,
    build_vocab_masks,
    find_question_premise_offsets,
    VocabMasks,
)
from .cache_utils import (
    kv_pairs,
    make_cache,
    slice_branch_kv,
    crop_cache,
    repeat_cache,
    select_batch_cache,
)
from .tree_builder import (
    pack_guided_tree,
    compute_question_premise_ll,
)
from .constrained import (
    encode_candidate,
    finalize_constrained_dict,
    score_candidates_packed,
    score_all_constrained_fused,
)
from .open_mode import (
    complete_batch_branched,
    complete_batch,
)
from .diagnostics import (
    tok_str,
    tok_flags,
    tokenization_line,
    format_open_diag,
    format_constrained_diag,
)
from .engine import (
    TreeEngine,
    init_engine,
    get_engine,
    predict_unified_tree,
    model_dtype,
    precision,
)
from .verification import (
    logit_metrics,
    tree_vs_sequential,
    verify_tree_equivalence,
    reference_candidate_logjoint,
    reference_complete,
    verify_stage2_equivalence,
    diagnose,
    diagnose_open,
)
from .calibration import (
    CALIBRATION_CASES,
    calibrate_ood_threshold,
)
from .suites import (
    SCENARIOS,
    HELDOUT,
)
from .harness import (
    display_evaluation,
    evaluate_scenarios,
    summarize,
)

__all__ = [
    # Engine & Global State
    "TreeEngine",
    "init_engine",
    "get_engine",
    "predict_unified_tree",
    "model_dtype",
    "precision",
    # Constants
    "P_STOP_FLOOR",
    "DEFAULT_OOD_MASS_THRESHOLD",
    "DEFAULT_OOD_PREMISE_THRESHOLD",
    "SUBWORD_BRANCH_FACTOR",
    "SUBWORD_BRANCH_RATIO",
    "DIAG_MODE",
    "KNOWN_BOUND_PREFIXES",
    "IRREGULAR_STEM_EXCEPTIONS",
    "GRAMMAR_GLUE",
    "QUESTION_STOPWORDS",
    "GUIDED_ANCHOR_PROMPT",
    # Lexicon
    "stem_word",
    "is_bound_bpe_prefix",
    "build_vocab_masks",
    "find_question_premise_offsets",
    "VocabMasks",
    # Cache
    "kv_pairs",
    "make_cache",
    "slice_branch_kv",
    "crop_cache",
    "repeat_cache",
    "select_batch_cache",
    # Tree builder
    "pack_guided_tree",
    "compute_question_premise_ll",
    # Constrained
    "encode_candidate",
    "finalize_constrained_dict",
    "score_candidates_packed",
    "score_all_constrained_fused",
    # Open mode
    "complete_batch_branched",
    "complete_batch",
    # Diagnostics
    "tok_str",
    "tok_flags",
    "tokenization_line",
    "format_open_diag",
    "format_constrained_diag",
    # Verification
    "logit_metrics",
    "tree_vs_sequential",
    "verify_tree_equivalence",
    "reference_candidate_logjoint",
    "reference_complete",
    "verify_stage2_equivalence",
    "diagnose",
    "diagnose_open",
    # Calibration & Suites
    "CALIBRATION_CASES",
    "calibrate_ood_threshold",
    "SCENARIOS",
    "HELDOUT",
    # Harness
    "display_evaluation",
    "evaluate_scenarios",
    "summarize",
]
