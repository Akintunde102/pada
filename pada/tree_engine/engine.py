"""Core TreeEngine encapsulation and evaluation pipeline."""

import contextlib
import math
import time
from typing import Any, Dict, Optional, Tuple, List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from .constants import (
    DEFAULT_OOD_MASS_THRESHOLD,
    DEFAULT_OOD_PREMISE_THRESHOLD,
    DIAG_MODE,
    P_STOP_FLOOR,
)
from .lexicon import (
    build_vocab_masks,
    stem_word,
    VocabMasks,
)
from .cache_utils import slice_branch_kv
from .tree_builder import pack_guided_tree, compute_question_premise_ll
from .constrained import (
    score_all_constrained_fused,
    finalize_constrained_dict,
)
from .open_mode import complete_batch_branched, complete_all_open_fused
from .diagnostics import (
    tok_str,
    tok_flags,
    format_open_diag,
    format_constrained_diag,
)


class TreeEngine:
    """Encapsulated Unified Tree Attention & Inference Engine."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        ood_mass_threshold: float = DEFAULT_OOD_MASS_THRESHOLD,
        ood_premise_threshold: float = DEFAULT_OOD_PREMISE_THRESHOLD,
        diag_mode: str = DIAG_MODE,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device.type
        self.dtype = dtype or next(model.parameters()).dtype
        self.ood_mass_threshold = ood_mass_threshold
        self.ood_premise_threshold = ood_premise_threshold
        self.diag_mode = diag_mode

        vocab_size = max(
            getattr(getattr(model, "config", None), "vocab_size", len(tokenizer)),
            len(tokenizer),
        )
        self.masks: VocabMasks = build_vocab_masks(tokenizer, vocab_size, device=self.device)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "TreeEngine":
        """Load pretrained model with flash/sdpa attention support."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            if device.startswith("cuda") and torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability()
                dtype = torch.bfloat16 if (major >= 8 and torch.cuda.is_bf16_supported()) else torch.float16
            else:
                dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs = {"low_cpu_mem_usage": True, "attn_implementation": "sdpa"}
        try:
            m = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **load_kwargs)
        except TypeError:
            try:
                m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **load_kwargs)
            except Exception:
                load_kwargs.pop("attn_implementation", None)
                m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **load_kwargs)

        m = m.to(device)
        m.eval()
        return cls(model=m, tokenizer=tokenizer, device=device, dtype=dtype, **kwargs)

    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @contextlib.contextmanager
    def precision(self, target: torch.dtype):
        """Temporarily cast model weights for exact float32 verification."""
        orig = self.model_dtype()
        if orig != target:
            self.model.to(target)
        try:
            yield
        finally:
            if orig != target:
                self.model.to(orig)

    def predict(
        self,
        state: str,
        questions: Dict[str, Any],
        top_k: int = 5,
        open_beam: int = 10,
    ) -> Dict[str, Any]:
        """Execute unified tree inference across open and constrained questions."""
        t0 = time.perf_counter()
        ids_t, mask_4d, pos_t, branch_meta, trunk_len, tree_seg = pack_guided_tree(
            state=state,
            questions=questions,
            tokenizer=self.tokenizer,
            id_to_word=self.masks.id_to_word,
            device=self.device,
            dtype=self.model_dtype(),
        )

        with torch.inference_mode():
            outputs = self.model(
                input_ids=ids_t,
                attention_mask=mask_4d,
                position_ids=pos_t,
                use_cache=True,
            )

        # Mode B: Score all constrained questions in a single fused Stage-2 pass
        constrained_specs = {
            q_key: q_spec["candidates"]
            for q_key, q_spec in questions.items()
            if isinstance(q_spec, dict) and q_spec.get("candidates")
        }

        fused_constrained = score_all_constrained_fused(
            constrained_specs=constrained_specs,
            tree_logits=outputs.logits,
            tree_kv=outputs.past_key_values,
            trunk_len=trunk_len,
            tree_seg=tree_seg,
            branch_meta=branch_meta,
            model=self.model,
            tokenizer=self.tokenizer,
            stop_boundary_mask=self.masks.stop_boundary_mask,
            device=self.device,
            p_stop_floor=P_STOP_FLOOR,
            tok_str_fn=lambda tid: tok_str(tid, self.tokenizer),
        )

        # Mode A: Prepare top-k first tokens for all Open Mode questions and score in a fused Stage-2 pass
        open_first_tids: Dict[str, List[int]] = {}
        open_logp_cache: Dict[str, torch.Tensor] = {}
        open_content_mass: Dict[str, float] = {}
        open_traces: Optional[Dict[str, List[Dict[str, Any]]]] = (
            {} if self.diag_mode != "off" else None
        )
        for q_key, meta in branch_meta.items():
            q_spec = questions[q_key]
            candidates = q_spec.get("candidates") if isinstance(q_spec, dict) else None
            if not candidates:
                branch_logits = outputs.logits[0, meta["readout_idx"], :]
                full_logp = F.log_softmax(branch_logits.float(), dim=-1)
                open_logp_cache[q_key] = full_logp
                open_content_mass[q_key] = float(
                    torch.exp(full_logp[self.masks.content_word_mask]).sum().item()
                )
                masked = branch_logits.float().masked_fill(
                    ~self.masks.content_word_mask, float("-inf")
                )
                open_first_tids[q_key] = torch.topk(masked, k=open_beam).indices.tolist()
                if open_traces is not None:
                    open_traces[q_key] = []

        fused_open = complete_all_open_fused(
            open_first_tids=open_first_tids,
            tree_kv=outputs.past_key_values,
            trunk_len=trunk_len,
            tree_seg=tree_seg,
            branch_meta=branch_meta,
            model=self.model,
            tokenizer=self.tokenizer,
            masks=self.masks,
            traces=open_traces,
            device=self.device,
            tok_str_fn=lambda tid: tok_str(tid, self.tokenizer),
            tok_flags_fn=lambda tid: tok_flags(tid, self.masks),
        )

        results: Dict[str, Any] = {}
        for q_key, meta in branch_meta.items():
            q_spec = questions[q_key]
            candidates = q_spec.get("candidates") if isinstance(q_spec, dict) else None
            expected = q_spec.get("expected") if isinstance(q_spec, dict) else None
            branch_logits = outputs.logits[0, meta["readout_idx"], :]

            if candidates:  # MODE B: CONSTRAINED
                lj_dict, det_dict = fused_constrained[q_key]
                premise_ll = compute_question_premise_ll(outputs.logits, meta)
                res = finalize_constrained_dict(
                    candidates=candidates,
                    log_joints=lj_dict,
                    details=det_dict,
                    premise_ll=premise_ll,
                    ood_mass_threshold=self.ood_mass_threshold,
                    ood_premise_threshold=self.ood_premise_threshold,
                )
                is_match = bool(expected) and (stem_word(res["word"]) == stem_word(expected) or res["word"] == expected)
                results[q_key] = {
                    "type": "CONSTRAINED",
                    "word": res["word"],
                    "confidence": res["confidence"],
                    "raw_mass": res["raw_mass"],
                    "log_raw_mass": res["log_raw_mass"],
                    "premise_ll": res["premise_ll"],
                    "is_ood": res["is_ood"],
                    "runner_up": res["runner_up"],
                    "expected": expected,
                    "correct": is_match,
                    "probabilities": res["probabilities"],
                }
                if self.diag_mode == "all" or (self.diag_mode == "fail" and expected and not is_match):
                    results[q_key]["diag"] = format_constrained_diag(
                        q_key=q_key,
                        expected=expected,
                        res=res,
                        full_logp=F.log_softmax(branch_logits.float(), dim=-1),
                        tokenizer=self.tokenizer,
                        masks=self.masks,
                    )
            else:  # MODE A: OPEN
                full_logp = open_logp_cache[q_key]
                content_mass = open_content_mass[q_key]
                path_records = fused_open[q_key]
                trace = open_traces[q_key] if open_traces is not None else None

                # Merge surface forms sharing a stem; never let a bare bound prefix override a full word
                concepts: Dict[str, Any] = {}
                for rec in path_records:
                    tid, w, p, clj = rec["first_tid"], rec["word"], rec["pieces"], rec["comp_lj"]
                    is_trunc = rec["is_bound_trunc"]
                    joint = math.exp(float(full_logp[tid].item()) + clj)
                    stem = stem_word(w)
                    rep_score = (0 if is_trunc else 1, joint)
                    if stem not in concepts:
                        concepts[stem] = {
                            "word": w,
                            "pieces": p,
                            "joint": joint,
                            "best_score": rep_score,
                            "members": [(w, joint)],
                        }
                    else:
                        concepts[stem]["joint"] += joint
                        concepts[stem]["members"].append((w, joint))
                        if rep_score > concepts[stem]["best_score"]:
                            concepts[stem].update(word=w, pieces=p, best_score=rep_score)

                raw_mass = sum(c["joint"] for c in concepts.values())
                ranked = sorted(concepts.values(), key=lambda x: x["joint"], reverse=True)
                winner = ranked[0]
                norm = raw_mass or 1e-12
                runner_up = (ranked[1]["word"], round(ranked[1]["joint"] / norm, 4)) if len(ranked) > 1 else None
                is_match = bool(expected) and (stem_word(winner["word"]) == stem_word(expected) or winner["word"] == expected)

                results[q_key] = {
                    "type": "OPEN",
                    "word": winner["word"],
                    "pieces": winner["pieces"],
                    "confidence": round(winner["joint"] / norm, 4),
                    "raw_mass": raw_mass,
                    "content_mass": round(content_mass, 4),
                    "premise_ll": None,
                    "is_ood": None,
                    "runner_up": runner_up,
                    "expected": expected,
                    "correct": is_match,
                    "top_k": {c["word"]: round(c["joint"] / norm, 4) for c in ranked[:top_k]},
                }
                if self.diag_mode == "all" or (self.diag_mode == "fail" and expected and not is_match):
                    results[q_key]["diag"] = format_open_diag(
                        q_key=q_key,
                        expected=expected,
                        winner=winner,
                        path_records=path_records,
                        full_logp=full_logp,
                        trace=trace,
                        concepts=concepts,
                        tokenizer=self.tokenizer,
                        masks=self.masks,
                    )

        del outputs

        return {
            "results": results,
            "metrics": {
                "trunk_tokens": trunk_len,
                "total_packed_tokens": ids_t.shape[1],
                "num_questions": len(questions),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
        }


# Global singleton instance for functional API compatibility
_ACTIVE_ENGINE: Optional[TreeEngine] = None


def init_engine(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> TreeEngine:
    """Initialize or inject the active TreeEngine singleton."""
    global _ACTIVE_ENGINE
    if model is None:
        _ACTIVE_ENGINE = TreeEngine.from_pretrained(model_name=model_name, device=device, dtype=dtype, **kwargs)
    else:
        _ACTIVE_ENGINE = TreeEngine(model=model, tokenizer=tokenizer, device=device, dtype=dtype, **kwargs)
    return _ACTIVE_ENGINE


def get_engine() -> TreeEngine:
    """Get active TreeEngine singleton, raising if not initialized."""
    if _ACTIVE_ENGINE is None:
        raise RuntimeError("TreeEngine is not initialized. Call init_engine() first.")
    return _ACTIVE_ENGINE


def predict_unified_tree(state: str, questions: Dict[str, Any], top_k: int = 5, open_beam: int = 10) -> Dict[str, Any]:
    """Execute prediction using the active singleton engine."""
    return get_engine().predict(state=state, questions=questions, top_k=top_k, open_beam=open_beam)


def model_dtype() -> torch.dtype:
    """Return active model dtype."""
    return get_engine().model_dtype()


def precision(target: torch.dtype):
    """Context manager for precision casting on active model."""
    return get_engine().precision(target)
