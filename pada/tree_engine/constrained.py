"""Constrained candidate evaluation (single-branch packed and multi-question fused scoring)."""

import math
from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from .constants import (
    P_STOP_FLOOR,
    DEFAULT_OOD_MASS_THRESHOLD,
    DEFAULT_OOD_PREMISE_THRESHOLD,
)
from .cache_utils import crop_cache


def encode_candidate(c: str, tokenizer: Any) -> List[int]:
    """Encode candidate word with leading space."""
    return tokenizer.encode(" " + c.replace("_", " ").strip(), add_special_tokens=False)


def finalize_constrained_dict(
    candidates: List[str],
    log_joints: Dict[str, float],
    details: Dict[str, Any],
    premise_ll: Optional[float] = None,
    ood_mass_threshold: float = DEFAULT_OOD_MASS_THRESHOLD,
    ood_premise_threshold: float = DEFAULT_OOD_PREMISE_THRESHOLD,
) -> Dict[str, Any]:
    """Normalize log joint likelihoods into probabilities and evaluate OOD thresholds."""
    mx = max(log_joints.values())
    log_raw_mass = mx + math.log(sum(math.exp(v - mx) for v in log_joints.values()))
    raw_mass = math.exp(log_raw_mass)
    prob_dict = {c_: round(math.exp(v - log_raw_mass), 4) for c_, v in log_joints.items()}
    sorted_c = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    winner = sorted_c[0][0]

    if premise_ll is not None:
        is_ood = (raw_mass < ood_mass_threshold) or (premise_ll < ood_premise_threshold)
    else:
        is_ood = raw_mass < ood_mass_threshold

    return {
        "word": winner,
        "confidence": sorted_c[0][1],
        "runner_up": sorted_c[1] if len(sorted_c) > 1 else None,
        "raw_mass": raw_mass,
        "log_raw_mass": log_raw_mass,
        "premise_ll": premise_ll,
        "is_ood": is_ood,
        "probabilities": prob_dict,
        "log_joints": log_joints,
        "details": details,
    }


def score_candidates_packed(
    candidates: List[str],
    branch_logits: torch.Tensor,
    isolated_kv: DynamicCache,
    model: Any,
    tokenizer: Any,
    stop_boundary_mask: torch.Tensor,
    device: Any = "cpu",
    p_stop_floor: float = P_STOP_FLOOR,
    tok_str_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Single-branch packed leaf scorer (used for Stage-2 isolated-cache equivalence verification)."""
    if tok_str_fn is None:
        tok_str_fn = lambda tid: tokenizer.convert_ids_to_tokens(int(tid))

    P = isolated_kv.get_seq_length()
    cand_tokens = [encode_candidate(c, tokenizer) for c in candidates]
    lens = [len(t) for t in cand_tokens]
    T = sum(lens)

    flat = [t for toks in cand_tokens for t in toks]
    seg = [i for i, l in enumerate(lens) for _ in range(l)]
    offs = [j for l in lens for j in range(l)]
    input_ids = torch.tensor([flat], dtype=torch.long, device=device)
    pos_ids = (P + torch.tensor(offs, dtype=torch.long, device=device)).unsqueeze(0)
    seg_t = torch.tensor(seg, dtype=torch.long, device=device)

    r = torch.arange(T, device=device).unsqueeze(1)
    c = torch.arange(T, device=device).unsqueeze(0)
    new_allowed = (seg_t.unsqueeze(1) == seg_t.unsqueeze(0)) & (c <= r)
    allowed = torch.cat([torch.ones((T, P), dtype=torch.bool, device=device), new_allowed], dim=1)
    mdt = next(model.parameters()).dtype
    mask = torch.zeros((1, 1, T, P + T), dtype=mdt, device=device)
    mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(mdt).min)

    try:
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=pos_ids,
                past_key_values=isolated_kv,
                use_cache=True,
            )
    finally:
        crop_cache(isolated_kv, P)

    logp = F.log_softmax(out.logits[0].float(), dim=-1)
    step1 = F.log_softmax(branch_logits.float(), dim=-1)

    log_joints, details = {}, {}
    start = 0
    for cand, toks in zip(candidates, cand_tokens):
        m = len(toks)
        lj = float(step1[toks[0]].item())
        for j in range(m - 1):
            lj += float(logp[start + j, toks[j + 1]].item())
        p_stop = float(torch.exp(logp[start + m - 1][stop_boundary_mask]).sum().item())
        lj += math.log(max(p_stop, p_stop_floor))
        log_joints[cand] = lj
        details[cand] = {
            "pieces": [tok_str_fn(t) for t in toks],
            "p_first": math.exp(float(step1[toks[0]].item())),
            "p_stop": p_stop,
            "log_joint": lj,
        }
        start += m

    return finalize_constrained_dict(candidates, log_joints, details)


def score_all_constrained_fused(
    constrained_specs: Dict[str, List[str]],
    tree_logits: torch.Tensor,
    tree_kv: DynamicCache,
    trunk_len: int,
    tree_seg: torch.Tensor,
    branch_meta: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    stop_boundary_mask: torch.Tensor,
    device: Any = "cpu",
    p_stop_floor: float = P_STOP_FLOOR,
    tok_str_fn: Optional[Any] = None,
) -> Dict[str, Tuple[Dict[str, float], Dict[str, Any]]]:
    """
    Packs all candidates for ALL constrained questions into ONE Stage-2 forward pass directly
    on top of `tree_kv` without slicing per-question KV caches. Cropped back to `P` in `finally`.
    """
    if not constrained_specs:
        return {}

    if tok_str_fn is None:
        tok_str_fn = lambda tid: tokenizer.convert_ids_to_tokens(int(tid))

    P = tree_kv.get_seq_length()
    flat_ids, pos_list, parent_branches, cand_ids = [], [], [], []
    layout = []

    cid = 0
    for q_key, candidates in constrained_specs.items():
        meta = branch_meta[q_key]
        b_idx = meta["branch_idx"]
        base_pos = trunk_len + meta["branch_len"]
        for cand in candidates:
            toks = encode_candidate(cand, tokenizer)
            start = len(flat_ids)
            flat_ids.extend(toks)
            pos_list.extend([base_pos + j for j in range(len(toks))])
            parent_branches.extend([b_idx] * len(toks))
            cand_ids.extend([cid] * len(toks))
            layout.append((q_key, cand, toks, start))
            cid += 1

    T = len(flat_ids)
    input_ids = torch.tensor([flat_ids], dtype=torch.long, device=device)
    pos_ids = torch.tensor([pos_list], dtype=torch.long, device=device)
    parent_t = torch.tensor(parent_branches, dtype=torch.long, device=device)
    cid_t = torch.tensor(cand_ids, dtype=torch.long, device=device)

    past_allowed = (tree_seg.unsqueeze(0) == 0) | (tree_seg.unsqueeze(0) == parent_t.unsqueeze(1))
    r = torch.arange(T, device=device).unsqueeze(1)
    c = torch.arange(T, device=device).unsqueeze(0)
    new_allowed = (cid_t.unsqueeze(1) == cid_t.unsqueeze(0)) & (c <= r)
    allowed = torch.cat([past_allowed, new_allowed], dim=1)

    mdt = next(model.parameters()).dtype
    mask = torch.zeros((1, 1, T, P + T), dtype=mdt, device=device)
    mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(mdt).min)

    try:
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=pos_ids,
                past_key_values=tree_kv,
                use_cache=True,
            )
    finally:
        crop_cache(tree_kv, P)

    logp = F.log_softmax(out.logits[0].float(), dim=-1)
    step1_cache = {
        q_key: F.log_softmax(tree_logits[0, branch_meta[q_key]["readout_idx"]].float(), dim=-1)
        for q_key in constrained_specs
    }

    per_q_lj = {q_key: {} for q_key in constrained_specs}
    per_q_det = {q_key: {} for q_key in constrained_specs}
    for q_key, cand, toks, start in layout:
        m = len(toks)
        step1 = step1_cache[q_key]
        lj = float(step1[toks[0]].item())
        for j in range(m - 1):
            lj += float(logp[start + j, toks[j + 1]].item())
        p_stop = float(torch.exp(logp[start + m - 1][stop_boundary_mask]).sum().item())
        lj += math.log(max(p_stop, p_stop_floor))
        per_q_lj[q_key][cand] = lj
        per_q_det[q_key][cand] = {
            "pieces": [tok_str_fn(t) for t in toks],
            "p_first": math.exp(float(step1[toks[0]].item())),
            "p_stop": p_stop,
            "log_joint": lj,
        }

    return {q_key: (per_q_lj[q_key], per_q_det[q_key]) for q_key in constrained_specs}
