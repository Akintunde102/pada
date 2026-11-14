"""Open-mode multi-token completion with bound-prefix guard and subword branching."""

import math
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from .constants import (
    P_STOP_FLOOR,
    SUBWORD_BRANCH_FACTOR,
    SUBWORD_BRANCH_RATIO,
    DIAG_TOP_NEXT,
)
from .cache_utils import repeat_cache, select_batch_cache, crop_cache
from .lexicon import VocabMasks


def complete_batch_branched(
    first_tids: List[int],
    isolated_kv: DynamicCache,
    model: Any,
    tokenizer: Any,
    masks: VocabMasks,
    max_tokens: int = 5,
    max_branches_per_root: int = SUBWORD_BRANCH_FACTOR,
    branch_ratio: float = SUBWORD_BRANCH_RATIO,
    forbid_bound_stop: bool = True,
    trace: Optional[List[Dict[str, Any]]] = None,
    device: Any = "cpu",
    p_stop_floor: float = P_STOP_FLOOR,
    diag_top_next: int = DIAG_TOP_NEXT,
    tok_str_fn: Optional[Any] = None,
    tok_flags_fn: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Extends every first token into complete words with:
      1. Bound-prefix guard (`forbid_bound_stop=True`): prevents truncated BPE subword stems
         from stopping at step 0 as non-words when a continuation is available.
      2. Subword branching at step 0: clones row when secondary continuation probability is high.
      3. Active-row batch pruning: finished rows are pruned from KV cache.
    """
    if tok_str_fn is None:
        tok_str_fn = lambda tid: tokenizer.convert_ids_to_tokens(int(tid))
    if tok_flags_fn is None:
        tok_flags_fn = lambda tid: "WORD" if masks.content_word_mask[int(tid)] else "other"

    K = len(first_tids)
    P = isolated_kv.get_seq_length()
    cache = repeat_cache(isolated_kv, K)

    paths = [
        {
            "path_id": i,
            "root_idx": i,
            "first_tid": t,
            "pieces": [masks.id_to_word.get(t, tokenizer.decode([t]).strip())],
            "logj": 0.0,
            "cur_tid": t,
        }
        for i, t in enumerate(first_tids)
    ]
    active_paths = list(paths)

    for s in range(max_tokens):
        if not active_paths:
            break
        B = len(active_paths)
        cur = torch.tensor([p["cur_tid"] for p in active_paths], dtype=torch.long, device=device).unsqueeze(1)
        pos = torch.full((B, 1), P + s, dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model(input_ids=cur, past_key_values=cache, position_ids=pos, use_cache=True)

        probs = F.softmax(out.logits[:, -1, :].float(), dim=-1)
        p_stop = probs[:, masks.stop_boundary_mask].sum(dim=1)
        cont = probs.masked_fill(~masks.subword_cont_mask, 0.0)
        cont_sum = cont.sum(dim=1)
        top_k_cont = max(1, max_branches_per_root if s == 0 else 1)
        best_p, best_t = cont.topk(k=top_k_cont, dim=1)

        next_active_paths = []
        keep_batch_rows = []

        for row_idx, path in enumerate(active_paths):
            ps = float(p_stop[row_idx].item())
            cs = float(cont_sum[row_idx].item())
            bp0 = float(best_p[row_idx, 0].item())
            bt0 = int(best_t[row_idx, 0].item())

            # A bound BPE prefix at step 0 (e.g. 'Ġemerg') MUST continue into a real word if possible
            is_bound_root = (
                forbid_bound_stop
                and s == 0
                and bool(masks.bound_prefix_mask[path["first_tid"]].item())
                and bp0 > 1e-6
            )
            should_stop = (not is_bound_root) and (ps >= cs or s == max_tokens - 1 or bp0 <= 0.0)

            if trace is not None:
                entry = {
                    "path_id": path["path_id"],
                    "row": path["root_idx"],
                    "step": s,
                    "p_stop": ps,
                    "p_cont": cs,
                    "p_other": max(0.0, 1.0 - ps - cs),
                    "stopped": should_stop,
                    "best_cont": (masks.id_to_word.get(bt0, "?"), bp0),
                    "top": [],
                }
                try:
                    tp, tt = probs[row_idx].topk(diag_top_next)
                    entry["top"] = [(tok_str_fn(t), float(p), tok_flags_fn(t)) for p, t in zip(tp.tolist(), tt.tolist())]
                except Exception as e:
                    entry["top"] = [(f"<trace error: {type(e).__name__}: {e}>", 0.0, "?")]
                trace.append(entry)

            if should_stop:
                path["logj"] += math.log(max(ps, p_stop_floor))
            else:
                if s == 0 and top_k_cont > 1:
                    for b in range(1, top_k_cont):
                        bp_alt = float(best_p[row_idx, b].item())
                        bt_alt = int(best_t[row_idx, b].item())
                        if bp_alt >= branch_ratio * bp0 and bp_alt > 1e-4:
                            alt_piece = masks.id_to_word.get(bt_alt, tokenizer.decode([bt_alt]).strip())
                            alt_path = {
                                "path_id": len(paths),
                                "root_idx": path["root_idx"],
                                "first_tid": path["first_tid"],
                                "pieces": list(path["pieces"]) + [alt_piece],
                                "logj": path["logj"] + math.log(max(bp_alt, 1e-12)),
                                "cur_tid": bt_alt,
                            }
                            paths.append(alt_path)
                            next_active_paths.append(alt_path)
                            keep_batch_rows.append(row_idx)

                piece0 = masks.id_to_word.get(bt0, tokenizer.decode([bt0]).strip())
                path["pieces"].append(piece0)
                path["logj"] += math.log(max(bp0, 1e-12))
                path["cur_tid"] = bt0
                next_active_paths.append(path)
                keep_batch_rows.append(row_idx)

        if not next_active_paths:
            break
        idx_t = torch.tensor(keep_batch_rows, dtype=torch.long, device=device)
        cache = select_batch_cache(cache, idx_t)
        active_paths = next_active_paths

    del cache
    return [
        {
            "path_id": p["path_id"],
            "root_idx": p["root_idx"],
            "first_tid": p["first_tid"],
            "word": "".join(p["pieces"]),
            "pieces": p["pieces"],
            "comp_lj": p["logj"],
            "is_bound_trunc": (len(p["pieces"]) == 1 and bool(masks.bound_prefix_mask[p["first_tid"]].item())),
        }
        for p in paths
    ]


def complete_batch(
    first_tids: List[int],
    isolated_kv: DynamicCache,
    model: Any,
    tokenizer: Any,
    masks: VocabMasks,
    max_tokens: int = 5,
    trace: Optional[List[Dict[str, Any]]] = None,
    device: Any = "cpu",
) -> Tuple[List[str], List[List[str]], List[float]]:
    """Unbranched wrapper (`forbid_bound_stop=False`) for exact 1-to-1 greedy reference equivalence."""
    recs = complete_batch_branched(
        first_tids=first_tids,
        isolated_kv=isolated_kv,
        model=model,
        tokenizer=tokenizer,
        masks=masks,
        max_tokens=max_tokens,
        max_branches_per_root=1,
        forbid_bound_stop=False,
        trace=trace,
        device=device,
    )
    return [r["word"] for r in recs], [r["pieces"] for r in recs], [r["comp_lj"] for r in recs]


def complete_all_open_fused(
    open_first_tids: Dict[str, List[int]],
    tree_kv: DynamicCache,
    trunk_len: int,
    tree_seg: torch.Tensor,
    branch_meta: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    masks: VocabMasks,
    max_tokens: int = 5,
    max_branches_per_root: int = SUBWORD_BRANCH_FACTOR,
    branch_ratio: float = SUBWORD_BRANCH_RATIO,
    forbid_bound_stop: bool = True,
    traces: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    device: Any = "cpu",
    p_stop_floor: float = P_STOP_FLOOR,
    diag_top_next: int = DIAG_TOP_NEXT,
    tok_str_fn: Optional[Any] = None,
    tok_flags_fn: Optional[Any] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fused multi-question Open-Mode completion directly on top of `tree_kv` (batch_size=1)
    without duplicating or slicing per-question KV caches.
    """
    if not open_first_tids:
        return {}

    if tok_str_fn is None:
        tok_str_fn = lambda tid: tokenizer.convert_ids_to_tokens(int(tid))
    if tok_flags_fn is None:
        tok_flags_fn = lambda tid: "WORD" if masks.content_word_mask[int(tid)] else "other"

    P = tree_kv.get_seq_length()
    mdt = next(model.parameters()).dtype
    bound_set = set(torch.nonzero(masks.bound_prefix_mask, as_tuple=False).view(-1).tolist())

    all_paths: Dict[str, List[Dict[str, Any]]] = {}
    active_paths: List[Dict[str, Any]] = []

    for q_key, first_tids in open_first_tids.items():
        meta = branch_meta[q_key]
        b_idx = meta["branch_idx"]
        base_pos = trunk_len + meta["branch_len"]
        q_paths = []
        for i, t in enumerate(first_tids):
            p = {
                "q_key": q_key,
                "b_idx": b_idx,
                "base_pos": base_pos,
                "path_id": i,
                "root_idx": i,
                "first_tid": t,
                "tok_ids": [t],
                "pieces": [masks.id_to_word.get(t, tokenizer.decode([t]).strip())],
                "logj": 0.0,
            }
            q_paths.append(p)
            active_paths.append(p)
        all_paths[q_key] = q_paths

    for s in range(max_tokens):
        if not active_paths:
            break

        flat_ids: List[int] = []
        pos_list: List[int] = []
        parent_branches: List[int] = []
        cid_list: List[int] = []
        readout_indices: List[int] = []

        for cid, p in enumerate(active_paths):
            toks = p["tok_ids"]
            m = len(toks)
            flat_ids.extend(toks)
            pos_list.extend([p["base_pos"] + j for j in range(m)])
            parent_branches.extend([p["b_idx"]] * m)
            cid_list.extend([cid] * m)
            readout_indices.append(len(flat_ids) - 1)

        T = len(flat_ids)
        input_ids = torch.tensor([flat_ids], dtype=torch.long, device=device)
        pos_ids = torch.tensor([pos_list], dtype=torch.long, device=device)
        parent_t = torch.tensor(parent_branches, dtype=torch.long, device=device)
        cid_t = torch.tensor(cid_list, dtype=torch.long, device=device)

        past_allowed = (tree_seg.unsqueeze(0) == 0) | (tree_seg.unsqueeze(0) == parent_t.unsqueeze(1))
        r = torch.arange(T, device=device).unsqueeze(1)
        c = torch.arange(T, device=device).unsqueeze(0)
        new_allowed = (cid_t.unsqueeze(1) == cid_t.unsqueeze(0)) & (c <= r)
        allowed = torch.cat([past_allowed, new_allowed], dim=1)

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

        readout_t = torch.tensor(readout_indices, dtype=torch.long, device=device)
        step_logits = out.logits[0, readout_t, :].float()
        probs = F.softmax(step_logits, dim=-1)
        p_stop_list = probs[:, masks.stop_boundary_mask].sum(dim=1).tolist()
        cont = probs.masked_fill(~masks.subword_cont_mask, 0.0)
        cont_sum_list = cont.sum(dim=1).tolist()
        top_k_cont = max(1, max_branches_per_root if s == 0 else 1)
        best_p, best_t = cont.topk(k=top_k_cont, dim=1)
        best_p_list = best_p.tolist()
        best_t_list = best_t.tolist()

        next_active_paths: List[Dict[str, Any]] = []

        for row_idx, path in enumerate(active_paths):
            ps = float(p_stop_list[row_idx])
            cs = float(cont_sum_list[row_idx])
            bp0 = float(best_p_list[row_idx][0])
            bt0 = int(best_t_list[row_idx][0])

            is_bound_root = (
                forbid_bound_stop
                and s == 0
                and (path["first_tid"] in bound_set)
                and bp0 > 1e-6
            )
            should_stop = (not is_bound_root) and (ps >= cs or s == max_tokens - 1 or bp0 <= 0.0)

            q_key = path["q_key"]
            if traces is not None and q_key in traces:
                entry = {
                    "path_id": path["path_id"],
                    "row": path["root_idx"],
                    "step": s,
                    "p_stop": ps,
                    "p_cont": cs,
                    "p_other": max(0.0, 1.0 - ps - cs),
                    "stopped": should_stop,
                    "best_cont": (masks.id_to_word.get(bt0, "?"), bp0),
                    "top": [],
                }
                try:
                    tp, tt = probs[row_idx].topk(diag_top_next)
                    entry["top"] = [(tok_str_fn(t), float(p), tok_flags_fn(t)) for p, t in zip(tp.tolist(), tt.tolist())]
                except Exception as e:
                    entry["top"] = [(f"<trace error: {type(e).__name__}: {e}>", 0.0, "?")]
                traces[q_key].append(entry)

            if should_stop:
                path["logj"] += math.log(max(ps, p_stop_floor))
            else:
                if s == 0 and top_k_cont > 1:
                    for b in range(1, top_k_cont):
                        bp_alt = float(best_p_list[row_idx][b])
                        bt_alt = int(best_t_list[row_idx][b])
                        if bp_alt >= branch_ratio * bp0 and bp_alt > 1e-4:
                            alt_piece = masks.id_to_word.get(bt_alt, tokenizer.decode([bt_alt]).strip())
                            q_paths = all_paths[q_key]
                            alt_path = {
                                "q_key": q_key,
                                "b_idx": path["b_idx"],
                                "base_pos": path["base_pos"],
                                "path_id": len(q_paths),
                                "root_idx": path["root_idx"],
                                "first_tid": path["first_tid"],
                                "tok_ids": list(path["tok_ids"]) + [bt_alt],
                                "pieces": list(path["pieces"]) + [alt_piece],
                                "logj": path["logj"] + math.log(max(bp_alt, 1e-12)),
                            }
                            q_paths.append(alt_path)
                            next_active_paths.append(alt_path)

                piece0 = masks.id_to_word.get(bt0, tokenizer.decode([bt0]).strip())
                path["tok_ids"].append(bt0)
                path["pieces"].append(piece0)
                path["logj"] += math.log(max(bp0, 1e-12))
                next_active_paths.append(path)

        active_paths = next_active_paths

    return {
        q_key: [
            {
                "path_id": p["path_id"],
                "root_idx": p["root_idx"],
                "first_tid": p["first_tid"],
                "word": "".join(p["pieces"]),
                "pieces": p["pieces"],
                "comp_lj": p["logj"],
                "is_bound_trunc": (len(p["pieces"]) == 1 and (p["first_tid"] in bound_set)),
            }
            for p in q_paths
        ]
        for q_key, q_paths in all_paths.items()
    }
