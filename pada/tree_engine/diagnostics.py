"""Investigative diagnostics and token inspection formatters."""

import math
from typing import Any, Dict, List
import torch

from .constants import DIAG_TOP_FIRST, DIAG_TOP_NEXT
from .lexicon import stem_word, VocabMasks


def tok_str(tid: Any, tokenizer: Any) -> str:
    """Convert token id to string representation."""
    return tokenizer.convert_ids_to_tokens(int(tid))


def tok_flags(tid: Any, masks: VocabMasks) -> str:
    """Format token classification flags (WORD, BOUND_PFX, CONT, STOP)."""
    tid = int(tid)
    f = []
    if masks.content_word_mask[tid]:
        f.append("WORD")
    if masks.bound_prefix_mask[tid]:
        f.append("BOUND_PFX")
    if masks.subword_cont_mask[tid]:
        f.append("CONT")
    if masks.stop_boundary_mask[tid]:
        f.append("STOP")
    return "+".join(f) if f else "other"


def tokenization_line(word: str, tokenizer: Any, masks: VocabMasks) -> str:
    """Format the token pieces and flags of a word with a leading space."""
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return " + ".join(f"{tok_str(t, tokenizer)!r}<{tok_flags(t, masks)}>" for t in ids)


def format_open_diag(
    q_key: str,
    expected: Any,
    winner: Dict[str, Any],
    path_records: List[Dict[str, Any]],
    full_logp: torch.Tensor,
    trace: List[Dict[str, Any]],
    concepts: Dict[str, Any],
    tokenizer: Any,
    masks: VocabMasks,
    top_first: int = DIAG_TOP_FIRST,
) -> str:
    """Format open mode branch diagnostic report."""
    L = [f"  ┌─ DIAG open:{q_key}   expected={expected!r}   winner={winner['word']!r}   pieces={winner['pieces']}"]
    if expected:
        L.append(f"  │ expected ' {expected}' tokenizes as: {tokenization_line(expected, tokenizer, masks)}")
    L.append(f"  │ winner   ' {winner['word']}' tokenizes as: {tokenization_line(winner['word'], tokenizer, masks)}")
    L.append("  │ stem groups (total joint; * = representative word):")
    for stem, c in sorted(concepts.items(), key=lambda kv: -kv[1]["joint"])[:6]:
        mem = ", ".join(f"{w}={j:.4f}{'*' if w == c['word'] else ''}" for w, j in c["members"])
        L.append(f"  │    {stem!r}: total={c['joint']:.4f}  [{mem}]")
    L.append(f"  │ branched paths expanded (showing top {top_first} root tokens):")
    shown_roots = set()
    for rec in path_records:
        r_idx = rec["root_idx"]
        if len(shown_roots) >= top_first and r_idx not in shown_roots:
            continue
        shown_roots.add(r_idx)
        tid, w, p, l = rec["first_tid"], rec["word"], rec["pieces"], rec["comp_lj"]
        pf = math.exp(float(full_logp[tid]))
        L.append(
            f"  │  root#{r_idx + 1} {tok_str(tid, tokenizer)!r}<{tok_flags(tid, masks)}> P(first)={pf:.4f} -> "
            f"word={w!r} pieces={p} joint={math.exp(float(full_logp[tid]) + l):.5f} stem={stem_word(w)!r}"
        )
        if trace:
            for t in trace:
                if t["path_id"] != rec["path_id"]:
                    continue
                nxt = ", ".join(f"{tok!r}({pr:.3f},{fl})" for tok, pr, fl in t["top"])
                verdict = "STOP" if t["stopped"] else "continue"
                L.append(
                    f"  │       step {t['step']}: P(stop)={t['p_stop']:.3f} P(cont)={t['p_cont']:.3f} "
                    f"P(other)={t['p_other']:.3f} -> {verdict}   model's top next tokens: {nxt}"
                )
    L.append("  └─")
    return "\n".join(L)


def format_constrained_diag(
    q_key: str,
    expected: Any,
    res: Dict[str, Any],
    full_logp: torch.Tensor,
    tokenizer: Any,
    masks: VocabMasks,
    top_next: int = DIAG_TOP_NEXT,
) -> str:
    """Format constrained question candidate scoring diagnostic report."""
    L = [
        f"  ┌─ DIAG constrained:{q_key}   expected={expected!r}   winner={res['word']!r}   "
        f"raw_mass={res['raw_mass']:.3g}   premise_ll={res.get('premise_ll', 0.0):.2f}"
    ]
    if expected and expected not in res["probabilities"]:
        L.append("  │ NOTE: the expected label is not among the candidates")
    L.append("  │ candidate tokenization & scoring:")
    for c, d in res["details"].items():
        L.append(
            f"  │    {c:<16} {' + '.join(map(repr, d['pieces'])):<34} P(first)={d['p_first']:.4g}  "
            f"P(stop|end)={d['p_stop']:.4g}  logJoint={d['log_joint']:.2f}  "
            f"share={res['probabilities'][c] * 100:.1f}%"
        )
    top = torch.topk(full_logp, top_next)
    L.append(
        "  │ model's own top first tokens (unconstrained): "
        + ", ".join(
            f"{tok_str(t, tokenizer)!r}({math.exp(float(p)):.3f},{tok_flags(t, masks)})"
            for p, t in zip(top.values.tolist(), top.indices.tolist())
        )
    )
    L.append("  └─")
    return "\n".join(L)
