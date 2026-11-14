"""Equivalence verification proofs between tree attention and sequential execution."""

import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from .constants import P_STOP_FLOOR
from .engine import get_engine, TreeEngine
from .tree_builder import pack_guided_tree
from .cache_utils import slice_branch_kv
from .constrained import (
    score_candidates_packed,
    score_all_constrained_fused,
    encode_candidate,
)
from .open_mode import complete_batch


def logit_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    """Compute divergence metrics between two logit vectors."""
    a, b = a.float(), b.float()
    la, lb = F.log_softmax(a, -1), F.log_softmax(b, -1)
    return {
        "max_logit": (a - b).abs().max().item(),
        "max_logp": (la - lb).abs().max().item(),
        "kl": F.kl_div(la, lb, log_target=True, reduction="sum").item(),
        "top1": int(a.argmax()) == int(b.argmax()),
        "top5": set(a.topk(5).indices.tolist()) == set(b.topk(5).indices.tolist()),
    }


def tree_vs_sequential(state: str, questions: Dict[str, Any], engine: Optional[TreeEngine] = None) -> Dict[str, Any]:
    """Compare tree attention logits at readout against isolated sequential runs."""
    eng = engine or get_engine()
    ids_t, mask_4d, pos_t, meta, S, _ = pack_guided_tree(
        state=state,
        questions=questions,
        tokenizer=eng.tokenizer,
        id_to_word=eng.masks.id_to_word,
        device=eng.device,
        dtype=eng.model_dtype(),
    )
    with torch.inference_mode():
        tree = eng.model(input_ids=ids_t, attention_mask=mask_4d, position_ids=pos_t, use_cache=False)

    out = {}
    for q_key, m in meta.items():
        seq_ids = torch.cat([ids_t[0, :S], ids_t[0, m["branch_start"]:m["branch_end"] + 1]]).unsqueeze(0)
        with torch.inference_mode():
            seq = eng.model(input_ids=seq_ids, use_cache=False)
        out[q_key] = logit_metrics(tree.logits[0, m["readout_idx"]], seq.logits[0, -1])

    del tree
    return out


def verify_tree_equivalence(
    state: str,
    questions: Dict[str, Any],
    engine: Optional[TreeEngine] = None,
    atol_fp32: float = 1e-3,
    kl_noise: float = 0.02,
) -> bool:
    """Verify mathematical equivalence between tree attention and isolated sequential passes in float32."""
    eng = engine or get_engine()
    print("\n" + "=" * 100)
    print("[EQUIVALENCE 1/2] Tree attention vs isolated sequential passes — float32 PROOF")
    print("=" * 100)
    ok = True
    with eng.precision(torch.float32):
        for q_key, m in tree_vs_sequential(state, questions, eng).items():
            good = m["max_logit"] <= atol_fp32 and m["max_logp"] <= atol_fp32 and m["top1"] and m["top5"]
            ok &= good
            print(
                f"  • [{q_key:<15}] max|dlogit|={m['max_logit']:.2e}  max|dlogp|={m['max_logp']:.2e}  "
                f"KL={m['kl']:.2e}  top1={m['top1']}  top5={m['top5']}  -> {'MATCH' if good else 'FAIL'}"
            )
    if not ok:
        raise AssertionError("fp32 tree attention diverged from sequential execution")
    print(">> fp32 PROOF VERIFIED")

    if eng.model_dtype() != torch.float32:
        print(f"\n[noise check] native {eng.model_dtype()} (informational, KL <= {kl_noise}):")
        for q_key, m in tree_vs_sequential(state, questions, eng).items():
            flag = "ok" if m["kl"] <= kl_noise else "HIGH"
            print(f"  • [{q_key:<15}] max|dlogit|={m['max_logit']:.3f}  KL={m['kl']:.2e}  top1={m['top1']}  -> {flag}")
            if m["kl"] > kl_noise:
                raise AssertionError(f"{eng.model_dtype()} noise above KL budget for {q_key}")

    if eng.device == "cuda":
        torch.cuda.empty_cache()
    return True


def reference_candidate_logjoint(prefix_ids: List[int], cand_toks: List[int], engine: Optional[TreeEngine] = None) -> float:
    """Compute ground-truth candidate log joint probability via an unshared forward pass."""
    eng = engine or get_engine()
    ids = torch.tensor([prefix_ids + cand_toks], dtype=torch.long, device=eng.device)
    with torch.inference_mode():
        logp = F.log_softmax(eng.model(input_ids=ids, use_cache=False).logits[0].float(), dim=-1)
    L = len(prefix_ids)
    lj = sum(float(logp[L - 1 + j, t].item()) for j, t in enumerate(cand_toks))
    p_stop = float(torch.exp(logp[-1][eng.masks.stop_boundary_mask]).sum().item())
    return lj + math.log(max(p_stop, P_STOP_FLOOR))


def reference_complete(
    prefix_ids: List[int],
    first_tid: int,
    engine: Optional[TreeEngine] = None,
    max_tokens: int = 5,
) -> Tuple[List[str], float]:
    """Greedy sequential reference completion token by token."""
    eng = engine or get_engine()
    ids = list(prefix_ids) + [first_tid]
    pieces = [eng.masks.id_to_word[first_tid]]
    lj = 0.0
    for s in range(max_tokens):
        with torch.inference_mode():
            logits = eng.model(input_ids=torch.tensor([ids], device=eng.device), use_cache=False).logits[0, -1]
        probs = F.softmax(logits.float(), dim=-1)
        p_stop = float(probs[eng.masks.stop_boundary_mask].sum().item())
        cont = probs.masked_fill(~eng.masks.subword_cont_mask, 0.0)
        cont_sum = float(cont.sum().item())
        bp, bt = cont.max(dim=0)
        if p_stop >= cont_sum or s == max_tokens - 1:
            lj += math.log(max(p_stop, P_STOP_FLOOR))
            break
        tid = int(bt)
        pieces.append(eng.masks.id_to_word.get(tid, eng.tokenizer.decode([tid]).strip()))
        lj += math.log(max(float(bp), 1e-12))
        ids.append(tid)
    return pieces, lj


def verify_stage2_equivalence(
    state: str,
    questions: Dict[str, Any],
    engine: Optional[TreeEngine] = None,
    atol: float = 1e-3,
) -> bool:
    """Verify stage-2 candidate scoring & pruned completion against isolated sequential baselines."""
    eng = engine or get_engine()
    print("\n" + "=" * 100)
    print("[EQUIVALENCE 2/2] Fused multi-branch scoring, sliced scoring & active-pruned completion — float32")
    print("=" * 100)
    ok = True
    with eng.precision(torch.float32):
        ids_t, mask_4d, pos_t, meta, S, tree_seg = pack_guided_tree(
            state=state,
            questions=questions,
            tokenizer=eng.tokenizer,
            id_to_word=eng.masks.id_to_word,
            device=eng.device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            outputs = eng.model(input_ids=ids_t, attention_mask=mask_4d, position_ids=pos_t, use_cache=True)

        P_tree = outputs.past_key_values.get_seq_length()
        constrained_specs = {
            k: v["candidates"] for k, v in questions.items() if isinstance(v, dict) and v.get("candidates")
        }
        fused_res = score_all_constrained_fused(
            constrained_specs=constrained_specs,
            tree_logits=outputs.logits,
            tree_kv=outputs.past_key_values,
            trunk_len=S,
            tree_seg=tree_seg,
            branch_meta=meta,
            model=eng.model,
            tokenizer=eng.tokenizer,
            stop_boundary_mask=eng.masks.stop_boundary_mask,
            device=eng.device,
            p_stop_floor=P_STOP_FLOOR,
        )
        tree_idem = outputs.past_key_values.get_seq_length() == P_tree

        for q_key, m in meta.items():
            spec = questions[q_key]
            branch_logits = outputs.logits[0, m["readout_idx"]]
            prefix = ids_t[0, :S].tolist() + ids_t[0, m["branch_start"]:m["branch_end"] + 1].tolist()
            kv = slice_branch_kv(outputs.past_key_values, S, m["branch_start"], m["branch_end"], device=eng.device)
            P0 = kv.get_seq_length()

            if spec.get("candidates"):
                res = score_candidates_packed(
                    candidates=spec["candidates"],
                    branch_logits=branch_logits,
                    isolated_kv=kv,
                    model=eng.model,
                    tokenizer=eng.tokenizer,
                    stop_boundary_mask=eng.masks.stop_boundary_mask,
                    device=eng.device,
                )
                again = score_candidates_packed(
                    candidates=spec["candidates"],
                    branch_logits=branch_logits,
                    isolated_kv=kv,
                    model=eng.model,
                    tokenizer=eng.tokenizer,
                    stop_boundary_mask=eng.masks.stop_boundary_mask,
                    device=eng.device,
                )
                idem = kv.get_seq_length() == P0 and tree_idem and all(
                    abs(res["log_joints"][c] - again["log_joints"][c]) < 1e-6 for c in spec["candidates"]
                )
                fused_lj = fused_res[q_key][0]
                worst_sliced, worst_fused = 0.0, 0.0
                for c in spec["candidates"]:
                    ref = reference_candidate_logjoint(prefix, encode_candidate(c, eng.tokenizer), eng)
                    worst_sliced = max(worst_sliced, abs(res["log_joints"][c] - ref))
                    worst_fused = max(worst_fused, abs(fused_lj[c] - ref))
                good = worst_sliced <= atol and worst_fused <= atol and idem
                print(
                    f"  • [{q_key:<15}] CONSTRAINED  max|dSliced|= {worst_sliced:.2e}  "
                    f"max|dFused|={worst_fused:.2e}  idem={idem}  -> {'MATCH' if good else 'FAIL'}"
                )
            else:
                masked = branch_logits.float().masked_fill(~eng.masks.content_word_mask, float("-inf"))
                top = torch.topk(masked, k=3).indices.tolist()
                words, pcs, lj = complete_batch(
                    first_tids=top,
                    isolated_kv=kv,
                    model=eng.model,
                    tokenizer=eng.tokenizer,
                    masks=eng.masks,
                    device=eng.device,
                )
                worst, same = 0.0, True
                for t, p, l in zip(top, pcs, lj):
                    rp, rl = reference_complete(prefix, t, eng)
                    same &= rp == p
                    worst = max(worst, abs(rl - l))
                good = same and worst <= atol
                print(
                    f"  • [{q_key:<15}] OPEN         pieces-equal={same}  max|dlogJoint|={worst:.2e}  "
                    f"-> {'MATCH' if good else 'FAIL'}"
                )
            ok &= good
        del outputs

    if not ok:
        raise AssertionError("Stage-2 (fused/sliced scoring or completion) diverged from full forward")
    print(">> Stage-2 PROOF VERIFIED")
    return True


def diagnose(state: str, questions: Dict[str, Any], q_key: str, engine: Optional[TreeEngine] = None):
    """Run single-question diagnosis and print detailed token path diagnostics."""
    eng = engine or get_engine()
    prev = eng.diag_mode
    eng.diag_mode = "all"
    try:
        out = eng.predict(state, {q_key: questions[q_key]})
    finally:
        eng.diag_mode = prev

    ans = out["results"][q_key]
    print(f"[diagnose] {q_key}: prediction={ans['word']!r} expected={ans['expected']!r} correct={ans['correct']}")
    print(ans.get("diag", "(no diagnostics)"))


diagnose_open = diagnose
