"""
Phase 8: Jev Web Server (FastAPI) Powered by Qwen2.5-1.5B
---------------------------------------------------------
Features:
  1. Qwen2.5-1.5B Backbone (d = 1536) with Colab-trained LoRA adapter.
  2. Universal Continuous Rating Judge (0.0 to 1.0) with dynamic scale detection (e.g. 0 to 1, 1 to 5, 1 to 10).
  3. Phase 7 Two-Stage Serving with SHA-256 State KV Cache.
  4. Plain Qwen-1.5B text generator for synthetic scenario drafting.
"""

import copy
import hashlib
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Ensure project root is on sys.path so `pada.schema` imports cleanly
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pada.schema import (
    AnonymousQuestion,
    ChoiceQuestion,
    ChoiceResult,
    HostModelAdapter,
    JevRequest,
    JevResponse,
    NoulQuestion,
    NoulResult,
    ScoreQuestion,
    ScoreResult,
    Usage,
    calculate_output_tokens,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
SPECIAL_TOKENS = ["<|state|>", "<|q|>", "<|opt|>", "<|readout|>",
                  "<|noul|>", "<|choice|>", "<|score|>"]


# =====================================================================
# 1. READOUT HEADS (Compatible with Qwen-1.5B d=1536)
# =====================================================================
class NoulHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)

    def forward(self, h):
        return self.proj(self.norm(h)).squeeze(-1)


class ChoiceHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.scale = math.sqrt(d)

    def forward(self, h_readout, h_opts):
        q = self.W_q(self.norm(h_readout))
        k = self.W_k(self.norm(h_opts))
        return (k @ q) / self.scale


class UniversalScoreHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)

    def forward(self, h):
        return self.proj(self.norm(h)).squeeze(-1)


# =====================================================================
# 2. GLOBAL STATE & ENGINE INITIALIZATION
# =====================================================================
tokenizer = None
jev_model = None
heads = None
calibrated_T = {"noul": 1.50, "choice": 2.00, "score": 1.50}
generator_model = None
STATE_CACHE = {}


def load_all_models():
    global tokenizer, jev_model, heads, calibrated_T, generator_model

    print("[*] Loading Qwen2.5-1.5B Tokenizer & 7 Special Tokens...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    print("[*] Loading Qwen2.5-1.5B with Trained LoRA Adapter...", flush=True)
    base_model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    base_model.resize_token_embeddings(len(tokenizer))
    jev_model = PeftModel.from_pretrained(base_model, "checkpoints/jev_1_5b_lora_adapter")
    jev_model.eval()

    d = jev_model.config.hidden_size
    print(f"[*] Initializing Judges for d = {d}...", flush=True)
    heads = nn.ModuleDict({
        "noul": NoulHead(d),
        "choice": ChoiceHead(d),
        "score": UniversalScoreHead(d),
    })

    ckpt_path = "checkpoints/jev_1_5b_heads_and_T.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        heads.load_state_dict(ckpt["heads"])
        calibrated_T = ckpt.get("T", calibrated_T)
        print(f"[+] Loaded trained judges from {ckpt_path}!")
    heads.eval()
    print(f"    Temperatures: T_noul={calibrated_T['noul']:.2f}, T_choice={calibrated_T['choice']:.2f}, T_score={calibrated_T['score']:.2f}")

    print("[*] Loading Qwen-1.5B Text Generator for Synthetic Samples...", flush=True)
    generator_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    generator_model.eval()
    print("[+] Qwen2.5-1.5B Server ready to serve requests!", flush=True)


# =====================================================================
# 3. HELPER: DYNAMIC SCALE PARSER
# =====================================================================
def parse_scale(text: str):
    """Detects custom ranges like 'from 0 to 1', '1 to 10', or '0-100'. Default is (1.0, 5.0)."""
    if not text:
        return 1.0, 5.0
    match = re.search(r'(?:from\s+)?([0-9.]+)\s*(?:to|-)\s*([0-9.]+)', text, re.IGNORECASE)
    if match:
        try:
            min_v = float(match.group(1))
            max_v = float(match.group(2))
            if max_v > min_v:
                return min_v, max_v
        except ValueError:
            pass
    return 1.0, 5.0


# =====================================================================
# 4. TWO-STAGE KV SERVING ENGINE
# =====================================================================
def get_state_memory(state_text: str):
    key = hashlib.sha256(state_text.encode("utf-8")).hexdigest()
    hit = key in STATE_CACHE
    if not hit:
        state_ids = tokenizer.encode(f"<|state|> {state_text}", add_special_tokens=False)
        with torch.no_grad():
            out = jev_model(input_ids=torch.tensor([state_ids]), use_cache=True)
        STATE_CACHE[key] = (out.past_key_values, len(state_ids))
    return copy.deepcopy(STATE_CACHE[key][0]), STATE_CACHE[key][1], hit, key


def serialize_branch(branch: AnonymousQuestion) -> str:
    """Serializes an anonymous question branch into Jev special-token format (zero user keys)."""
    if branch.question_type == "noul":
        return f"<|q|> <|noul|> {branch.instructions} <|readout|>"
    elif branch.question_type == "choice":
        opts_str = " ".join(f"{opt.text} <|opt|>" for opt in branch.options)
        return f"<|q|> <|choice|> {branch.instructions} {opts_str} <|readout|>"
    elif branch.question_type == "score":
        rubric = ""
        if branch.score_criteria:
            items = [f"{lvl}: {desc}" for lvl, desc in sorted(branch.score_criteria.items())]
            rubric = " (" + "; ".join(items) + ")"
        return f"<|q|> <|score|> {branch.instructions}{rubric} <|readout|>"
    else:
        raise ValueError(f"Unsupported question type: {branch.question_type}")


def continuous_to_ordinal_distribution(norm_score: float, levels: int) -> Tuple[List[float], float]:
    """Maps continuous score p in (0, 1) to a K-level ordinal distribution over [0, ..., K-1]
    and computes E[S] = sum(k * P(S=k)).
    """
    p = min(max(float(norm_score), 1e-6), 1.0 - 1e-6)
    n = levels - 1
    dist: List[float] = []
    for k in range(levels):
        prob_k = math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
        dist.append(prob_k)
    total = sum(dist)
    dist = [x / total for x in dist]
    expected = sum(k * prob_k for k, prob_k in enumerate(dist))
    return dist, expected


def evaluate_jev_request(request: JevRequest) -> Tuple[JevResponse, Dict[str, Any]]:
    """Executes a full JevRequest through HostModelAdapter key stripping, Tree Attention,
    and readout heads, returning the canonical JevResponse and telemetry metadata.
    """
    t0 = time.time()

    # 1. Strip all user question keys and option keys on the host side
    stripped_input, mapping = HostModelAdapter.strip_request(request)

    # 2. Fetch or compute Stage 1 KV cache for the shared state
    cache, S, hit, state_hash = get_state_memory(stripped_input.state)

    # 3. Tokenize each anonymous branch and record per-branch token spans
    readout_id = tokenizer.convert_tokens_to_ids("<|readout|>")
    opt_id = tokenizer.convert_tokens_to_ids("<|opt|>")

    q_ids: List[int] = []
    q_seg: List[int] = []
    q_pos: List[int] = []
    branch_spans: Dict[int, Tuple[int, int]] = {}

    for k, branch in enumerate(stripped_input.branches, start=1):
        branch_text = serialize_branch(branch)
        piece = tokenizer.encode(branch_text, add_special_tokens=False)
        start_idx = len(q_ids)
        q_ids.extend(piece)
        end_idx = len(q_ids)
        branch_spans[branch.index] = (start_idx, end_idx)
        q_seg.extend([k] * len(piece))
        q_pos.extend(range(S, S + len(piece)))

    # 4. Build Tree Attention mask: [1, 1, Q, S + Q]
    Q = len(q_ids)
    can_see_state = torch.ones((Q, S), dtype=torch.bool)
    seg_t = torch.tensor(q_seg)
    rows, cols = torch.arange(Q).unsqueeze(1), torch.arange(Q).unsqueeze(0)
    can_see_q = (cols <= rows) & (seg_t.unsqueeze(1) == seg_t.unsqueeze(0))

    allowed = torch.cat([can_see_state, can_see_q], dim=1)
    mask = torch.zeros((1, 1, Q, S + Q)).masked_fill_(~allowed, float("-inf"))

    # 5. Single forward pass across all question branches
    with torch.no_grad():
        h_q = jev_model(
            input_ids=torch.tensor([q_ids]),
            attention_mask=mask,
            position_ids=torch.tensor([q_pos]),
            past_key_values=cache,
        ).last_hidden_state[0]

    # 6. Evaluate each branch dynamically using its readout and option indices
    raw_outputs: Dict[int, Any] = {}
    branch_norm_scores: Dict[int, float] = {}

    with torch.no_grad():
        for branch in stripped_input.branches:
            b_start, b_end = branch_spans[branch.index]
            b_readouts = [i for i in range(b_start, b_end) if q_ids[i] == readout_id]
            if not b_readouts:
                raise RuntimeError(f"Missing <|readout|> token in branch {branch.index}")
            r_idx = b_readouts[-1]

            if branch.question_type == "noul":
                p_noul = torch.sigmoid(heads["noul"](h_q[r_idx]) / calibrated_T["noul"]).item()
                raw_outputs[branch.index] = p_noul

            elif branch.question_type == "choice":
                b_opts = [i for i in range(b_start, b_end) if q_ids[i] == opt_id]
                q_vec = h_q[r_idx]
                o_vec = h_q[b_opts]
                c_probs = F.softmax(heads["choice"](q_vec, o_vec) / calibrated_T["choice"], -1).tolist()
                raw_outputs[branch.index] = c_probs

            elif branch.question_type == "score":
                levels = branch.score_levels or 5
                head_out = heads["score"](h_q[r_idx]) / calibrated_T["score"]
                if head_out.numel() == 1:
                    norm_score = torch.sigmoid(head_out).item()
                    branch_norm_scores[branch.index] = norm_score
                    dist, expected = continuous_to_ordinal_distribution(norm_score, levels)
                else:
                    probs_t = F.softmax(head_out, dim=-1)
                    dist = probs_t.tolist()
                    expected = sum(k * p for k, p in enumerate(dist))
                    branch_norm_scores[branch.index] = expected / max(1, levels - 1)
                raw_outputs[branch.index] = (dist, expected)

    # 7. Calculate synthetic token billing and reconstruct user-keyed JevResponse
    billed_output_tokens = calculate_output_tokens(request.questions, tokenizer=tokenizer)
    usage = Usage(input_tokens=S + Q, output_tokens=billed_output_tokens)
    jev_response = HostModelAdapter.reconstruct_response(mapping, raw_outputs, usage)

    latency_ms = (time.time() - t0) * 1000.0
    meta = {
        "latency_ms": latency_ms,
        "cache_hit": hit,
        "state_hash": state_hash,
        "tokens_s": S,
        "tokens_q": Q,
        "branch_norm_scores": branch_norm_scores,
    }
    return jev_response, meta


def evaluate_jev(state: str, noul_question: str, choice_question: str, options: List[str], score_question: str, score_scale: Optional[str] = None):
    """Studio UI helper that builds a canonical JevRequest and delegates to evaluate_jev_request."""
    criteria_map = {f"opt_{i}": opt_text for i, opt_text in enumerate(options)}
    jev_req = JevRequest(
        state=state,
        questions={
            "noul_q": NoulQuestion(instructions=noul_question),
            "choice_q": ChoiceQuestion(instructions=choice_question, criteria=criteria_map),
            "score_q": ScoreQuestion(instructions=score_question, levels=5),
        },
    )

    jev_res, meta = evaluate_jev_request(jev_req)

    noul_res = jev_res.results["noul_q"]
    choice_res = jev_res.results["choice_q"]
    score_res = jev_res.results["score_q"]

    assert isinstance(noul_res, NoulResult)
    assert isinstance(choice_res, ChoiceResult)
    assert isinstance(score_res, ScoreResult)

    c_probs = [choice_res.probabilities[f"opt_{i}"] for i in range(len(options))]
    best_idx = int(torch.tensor(c_probs).argmax())
    choice_all = [{"option": opt, "prob": prob} for opt, prob in zip(options, c_probs)]

    norm_score = meta["branch_norm_scores"].get(2, score_res.expected / 4.0)
    scale_text = score_scale if score_scale else score_question
    min_val, max_val = parse_scale(scale_text)
    scaled_rating = min_val + norm_score * (max_val - min_val)

    return {
        "latency_ms": meta["latency_ms"],
        "cache_hit": meta["cache_hit"],
        "state_hash": meta["state_hash"],
        "tokens_s": meta["tokens_s"],
        "tokens_q": meta["tokens_q"],
        "noul_prob": noul_res.p_true,
        "choice_winner": options[best_idx],
        "choice_winner_prob": c_probs[best_idx],
        "choice_all": choice_all,
        "normalized_score": norm_score,
        "expected_rating": scaled_rating,
        "score_distribution": score_res.distribution,
        "score_expected_0_indexed": score_res.expected,
        "scale_min": min_val,
        "scale_max": max_val,
        "jev_response": jev_res.model_dump(),
    }


# =====================================================================
# 5. FASTAPI APPLICATION
# =====================================================================
app = FastAPI(title="Jev Qwen-1.5B Evaluation Studio")


class EvaluateRequest(BaseModel):
    state: str
    noul_question: str
    choice_question: str
    options: List[str]
    score_question: str
    score_scale: Optional[str] = None


class GenerateRequest(BaseModel):
    prompt: str


@app.on_event("startup")
def startup_event():
    load_all_models()


@app.get("/", response_class=HTMLResponse)
def get_studio_ui():
    with open("scripts/studio_ui.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/v1/evaluate", response_model=JevResponse)
def api_v1_evaluate(req: JevRequest) -> JevResponse:
    """Canonical Jev / Pada evaluation endpoint complying strictly with pada.schema."""
    jev_res, _ = evaluate_jev_request(req)
    return jev_res


@app.post("/api/evaluate")
def api_evaluate(req: EvaluateRequest):
    return evaluate_jev(
        state=req.state,
        noul_question=req.noul_question,
        choice_question=req.choice_question,
        options=req.options,
        score_question=req.score_question,
        score_scale=req.score_scale,
    )


@app.post("/api/generate_sample")
def api_generate_sample(req: GenerateRequest):
    prompt_text = f"Customer review for a product ({req.prompt}):\n\""
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    with torch.no_grad():
        out = generator_model.generate(
            input_ids=input_ids,
            max_new_tokens=65,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_text = tokenizer.decode(out[0][len(input_ids[0]):], skip_special_tokens=True).strip()
    if gen_text.endswith('"'):
        gen_text = gen_text[:-1].strip()
    elif '"' in gen_text:
        gen_text = gen_text.split('"')[0].strip()
    return {"sample": gen_text}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
