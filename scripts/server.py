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
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

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
def parse_scale(question_text: str):
    """Detects custom ranges like 'from 0 to 1' or 'from 1 to 10'. Default is (1.0, 5.0)."""
    match = re.search(r'from\s+([0-9.]+)\s+to\s+([0-9.]+)', question_text, re.IGNORECASE)
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


def evaluate_jev(state: str, noul_question: str, choice_question: str, options: List[str], score_question: str):
    t0 = time.time()
    cache, S, hit, state_hash = get_state_memory(state)

    q_parts = [
        f"<|q|> <|noul|> {noul_question} <|readout|>",
        f"<|q|> <|choice|> {choice_question} " + " ".join(f"{o} <|opt|>" for o in options) + " <|readout|>",
        f"<|q|> <|score|> {score_question} <|readout|>",
    ]
    q_pieces = [tokenizer.encode(p, add_special_tokens=False) for p in q_parts]

    q_ids, q_seg, q_pos = [], [], []
    for k, piece in enumerate(q_pieces, start=1):
        q_ids += piece
        q_seg += [k] * len(piece)
        q_pos += list(range(S, S + len(piece)))

    Q = len(q_ids)
    can_see_state = torch.ones((Q, S), dtype=torch.bool)
    seg_t = torch.tensor(q_seg)
    rows, cols = torch.arange(Q).unsqueeze(1), torch.arange(Q).unsqueeze(0)
    can_see_q = (cols <= rows) & (seg_t.unsqueeze(1) == seg_t.unsqueeze(0))

    allowed = torch.cat([can_see_state, can_see_q], dim=1)
    mask = torch.zeros((1, 1, Q, S + Q)).masked_fill_(~allowed, float("-inf"))

    with torch.no_grad():
        h_q = jev_model(
            input_ids=torch.tensor([q_ids]),
            attention_mask=mask,
            position_ids=torch.tensor([q_pos]),
            past_key_values=cache,
        ).last_hidden_state[0]

    readout_id = tokenizer.convert_tokens_to_ids("<|readout|>")
    opt_id = tokenizer.convert_tokens_to_ids("<|opt|>")
    readouts = [i for i, t in enumerate(q_ids) if t == readout_id]
    opts = [i for i, t in enumerate(q_ids) if t == opt_id]

    q_vec = h_q[readouts[1]]
    o_vec = h_q[opts]

    with torch.no_grad():
        # 1. Noul (Calibrated Yes/No)
        p_noul = torch.sigmoid(heads["noul"](h_q[readouts[0]]) / calibrated_T["noul"]).item()

        # 2. Choice (Calibrated Option Probabilities)
        c_probs = F.softmax(heads["choice"](q_vec, o_vec) / calibrated_T["choice"], -1).tolist()

        # 3. Universal Continuous Score (Normalized 0.0 to 1.0)
        norm_score = torch.sigmoid(heads["score"](h_q[readouts[2]]) / calibrated_T["score"]).item()

    best_idx = int(torch.tensor(c_probs).argmax())
    choice_all = [{"option": opt, "prob": prob} for opt, prob in zip(options, c_probs)]

    # Dynamic Range Scaling
    min_val, max_val = parse_scale(score_question)
    scaled_rating = min_val + norm_score * (max_val - min_val)

    latency_ms = (time.time() - t0) * 1000.0

    return {
        "latency_ms": latency_ms,
        "cache_hit": hit,
        "state_hash": state_hash,
        "tokens_s": S,
        "tokens_q": Q,
        "noul_prob": p_noul,
        "choice_winner": options[best_idx],
        "choice_winner_prob": c_probs[best_idx],
        "choice_all": choice_all,
        "normalized_score": norm_score,
        "expected_rating": scaled_rating,
        "scale_min": min_val,
        "scale_max": max_val,
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


class GenerateRequest(BaseModel):
    prompt: str


@app.on_event("startup")
def startup_event():
    load_all_models()


@app.get("/", response_class=HTMLResponse)
def get_studio_ui():
    with open("scripts/studio_ui.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/evaluate")
def api_evaluate(req: EvaluateRequest):
    return evaluate_jev(
        state=req.state,
        noul_question=req.noul_question,
        choice_question=req.choice_question,
        options=req.options,
        score_question=req.score_question,
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
