"""Tree attention mask construction, sequence packing, and question premise log-likelihood."""

from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F

from .constants import GUIDED_ANCHOR_PROMPT
from .lexicon import find_question_premise_offsets


def pack_guided_tree(
    state: str,
    questions: Dict[str, Any],
    tokenizer: Any,
    id_to_word: Dict[int, str],
    device: Any = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any], int, torch.Tensor]:
    """
    Pack trunk prompt + state with isolated question branches into a 4D masked sequence.
    Returns:
        (ids_t, mask_4d, pos_t, branch_metadata, trunk_len, segments_tensor)
    """
    trunk_text = GUIDED_ANCHOR_PROMPT + f"Situation: {state.strip()}\n"
    trunk_ids = tokenizer.encode(trunk_text, add_special_tokens=False)
    S = len(trunk_ids)

    all_ids = list(trunk_ids)
    segments = [0] * S
    pos_ids = list(range(S))
    branch_metadata: Dict[str, Any] = {}

    for branch_idx, (q_key, q_spec) in enumerate(questions.items(), start=1):
        instr = q_spec["instructions"] if isinstance(q_spec, dict) else str(q_spec)
        branch_ids = tokenizer.encode(f"Question: {instr.strip()} One-word answer:", add_special_tokens=False)
        B_len = len(branch_ids)
        branch_start = len(all_ids)
        all_ids.extend(branch_ids)
        branch_end = len(all_ids) - 1
        segments.extend([branch_idx] * B_len)
        pos_ids.extend([S + i for i in range(B_len)])
        branch_metadata[q_key] = {
            "branch_idx": branch_idx,
            "branch_start": branch_start,
            "branch_end": branch_end,
            "readout_idx": branch_end,
            "branch_len": B_len,
            "branch_ids": branch_ids,
            "premise_offsets": find_question_premise_offsets(branch_ids, id_to_word, tokenizer),
        }

    total_len = len(all_ids)
    seg = torch.tensor(segments, dtype=torch.long, device=device)
    row = torch.arange(total_len, device=device).unsqueeze(1)
    col = torch.arange(total_len, device=device).unsqueeze(0)
    allowed = (col <= row) & ((seg.unsqueeze(0) == 0) | (seg.unsqueeze(1) == seg.unsqueeze(0)))

    mask = torch.zeros((1, 1, total_len, total_len), dtype=dtype, device=device)
    mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)

    ids_t = torch.tensor([all_ids], dtype=torch.long, device=device)
    pos_t = torch.tensor([pos_ids], dtype=torch.long, device=device)
    return ids_t, mask, pos_t, branch_metadata, S, seg


def compute_question_premise_ll(tree_logits: torch.Tensor, meta: Dict[str, Any]) -> float:
    """
    Zero-overhead OOD signal: computes the mean log-likelihood of the domain content tokens
    inside the question branch conditioned on the situation trunk.
    """
    b_start = meta["branch_start"]
    b_ids = meta["branch_ids"]
    offsets = meta["premise_offsets"]
    lls = []
    for off in offsets:
        if off == 0:
            continue
        pred_pos = b_start + off - 1
        target_tid = b_ids[off]
        logp = F.log_softmax(tree_logits[0, pred_pos].float(), dim=-1)
        lls.append(float(logp[target_tid].item()))
    return sum(lls) / max(len(lls), 1) if lls else 0.0
