"""
Phase 7 Educational Demo: Option A (Single Pass) vs. Option B (Two-Stage KV Cache)
-----------------------------------------------------------------------------------
Loads the Colab-trained checkpoint (LoRA + 3 Judges + Temperatures T) and proves:
  1. Option A and Option B produce the EXACT same thought vectors and probabilities.
  2. Option B reuses the cached <|state|> memory on repeated states (Cache Hit!).
"""

import copy
import hashlib
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
SPECIAL_TOKENS = ["<|state|>", "<|q|>", "<|opt|>", "<|readout|>",
                  "<|noul|>", "<|choice|>", "<|score|>"]

QUESTIONS = {
    "noul": "Is this customer review positive?",
    "choice": "What is the overall sentiment tier of this review?",
    "options": ["Negative Review", "Neutral Review", "Positive Review"],
    "score": "Rate overall satisfaction from 1 to 5:",
}


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


class ScoreHead(nn.Module):
    def __init__(self, d, levels=5):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, levels)

    def forward(self, h):
        return self.proj(self.norm(h))


def load_trained_jev():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    base_model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, "checkpoints/jev_lora_adapter")
    model.eval()

    d = model.config.hidden_size
    heads = nn.ModuleDict({"noul": NoulHead(d), "choice": ChoiceHead(d), "score": ScoreHead(d)})
    ckpt = torch.load("checkpoints/jev_heads_and_T.pt", map_location="cpu", weights_only=True)
    heads.load_state_dict(ckpt["heads"])
    heads.eval()

    return model, tokenizer, heads, ckpt["T"]


def extract_option_a(model, tokenizer, state_text):
    q = QUESTIONS
    parts = [
        f"<|state|> {state_text}",
        f"<|q|> <|noul|> {q['noul']} <|readout|>",
        f"<|q|> <|choice|> {q['choice']} " + " ".join(f"{o} <|opt|>" for o in q["options"]) + " <|readout|>",
        f"<|q|> <|score|> {q['score']} <|readout|>",
    ]
    pieces = [tokenizer.encode(p, add_special_tokens=False) for p in parts]

    ids, seg, pos = [], [], []
    S = len(pieces[0])
    for k, piece in enumerate(pieces):
        ids += piece
        seg += [k] * len(piece)
        start = 0 if k == 0 else S
        pos += list(range(start, start + len(piece)))

    n = len(ids)
    seg_t = torch.tensor(seg)
    rows, cols = torch.arange(n).unsqueeze(1), torch.arange(n).unsqueeze(0)
    allowed = (cols <= rows) & ((seg_t.unsqueeze(0) == 0) | (seg_t.unsqueeze(1) == seg_t.unsqueeze(0)))
    mask = torch.zeros((1, 1, n, n)).masked_fill_(~allowed, float("-inf"))

    with torch.no_grad():
        h = model(input_ids=torch.tensor([ids]), attention_mask=mask,
                  position_ids=torch.tensor([pos])).last_hidden_state[0]

    readout_id = tokenizer.convert_tokens_to_ids("<|readout|>")
    opt_id = tokenizer.convert_tokens_to_ids("<|opt|>")
    readouts = [i for i, t in enumerate(ids) if t == readout_id]
    opts = [i for i, t in enumerate(ids) if t == opt_id]

    return {
        "noul": h[readouts[0]],
        "choice_q": h[readouts[1]],
        "choice_opts": h[opts],
        "score": h[readouts[2]],
    }


STATE_CACHE = {}


def get_state_memory(model, tokenizer, state_text):
    key = hashlib.sha256(state_text.encode("utf-8")).hexdigest()
    hit = key in STATE_CACHE
    if not hit:
        state_ids = tokenizer.encode(f"<|state|> {state_text}", add_special_tokens=False)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([state_ids]), use_cache=True)
        STATE_CACHE[key] = (out.past_key_values, len(state_ids))
    return copy.deepcopy(STATE_CACHE[key][0]), STATE_CACHE[key][1], hit


def extract_option_b(model, tokenizer, state_text):
    cache, S, hit = get_state_memory(model, tokenizer, state_text)

    q = QUESTIONS
    q_parts = [
        f"<|q|> <|noul|> {q['noul']} <|readout|>",
        f"<|q|> <|choice|> {q['choice']} " + " ".join(f"{o} <|opt|>" for o in q["options"]) + " <|readout|>",
        f"<|q|> <|score|> {q['score']} <|readout|>",
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
        h_q = model(
            input_ids=torch.tensor([q_ids]),
            attention_mask=mask,
            position_ids=torch.tensor([q_pos]),
            past_key_values=cache,
        ).last_hidden_state[0]

    readout_id = tokenizer.convert_tokens_to_ids("<|readout|>")
    opt_id = tokenizer.convert_tokens_to_ids("<|opt|>")
    readouts = [i for i, t in enumerate(q_ids) if t == readout_id]
    opts = [i for i, t in enumerate(q_ids) if t == opt_id]

    vecs = {
        "noul": h_q[readouts[0]],
        "choice_q": h_q[readouts[1]],
        "choice_opts": h_q[opts],
        "score": h_q[readouts[2]],
    }
    return vecs, hit


def predict_calibrated(heads, T, vecs):
    q_vec = vecs["choice_q"]
    o_vec = vecs["choice_opts"]
    options = QUESTIONS["options"]
    with torch.no_grad():
        p = torch.sigmoid(heads["noul"](vecs["noul"]) / T["noul"]).item()
        c = F.softmax(heads["choice"](q_vec, o_vec) / T["choice"], -1)
        s = F.softmax(heads["score"](vecs["score"]) / T["score"], -1)
        expected = (torch.arange(1, 6).float() * s).sum().item()
    best = int(c.argmax())
    return p, options[best], c[best].item(), expected


def main():
    print("[1] Loading your Colab-trained checkpoint from checkpoints/...", flush=True)
    model, tokenizer, heads, T = load_trained_jev()
    print(f"    Loaded Temperatures: T_noul={T['noul']:.2f}, T_choice={T['choice']:.2f}, T_score={T['score']:.2f}")

    # A longer review so we can clearly see the speedup when the State is cached!
    long_state = (
        " ".join(["We bought this espresso machine for our busy office kitchen last month."] * 8)
        + " Overall, it is okay for the price. Works as expected, nothing fancy or special."
    )

    print("\n[2] Running Option A (Single Pass - reads State + Questions together)...", flush=True)
    t0 = time.time()
    vecs_a = extract_option_a(model, tokenizer, long_state)
    time_a = time.time() - t0
    pred_a = predict_calibrated(heads, T, vecs_a)

    print("[3] Running Option B - Request #1 (Cold Cache - computes & saves State KV)...", flush=True)
    t0 = time.time()
    vecs_b1, hit1 = extract_option_b(model, tokenizer, long_state)
    time_b1 = time.time() - t0
    pred_b1 = predict_calibrated(heads, T, vecs_b1)

    print("[4] Running Option B - Request #2 (Warm Cache Hit - skips State completely!)...", flush=True)
    t0 = time.time()
    vecs_b2, hit2 = extract_option_b(model, tokenizer, long_state)
    time_b2 = time.time() - t0
    pred_b2 = predict_calibrated(heads, T, vecs_b2)

    max_diff = max((vecs_a[k] - vecs_b2[k]).abs().max().item() for k in vecs_a)

    print("\n" + "=" * 76)
    print("PHASE 7 VERIFICATION: OPTION A vs. OPTION B (TWO-STAGE KV CACHE)")
    print("=" * 76)
    print(f"Max vector difference between Option A and Option B : {max_diff:.7f}")
    print(f"Option A prediction  : Positive={pred_a[0]*100:.1f}% | {pred_a[1]} ({pred_a[2]*100:.1f}%) | Rating={pred_a[3]:.2f}")
    print(f"Option B prediction  : Positive={pred_b2[0]*100:.1f}% | {pred_b2[1]} ({pred_b2[2]*100:.1f}%) | Rating={pred_b2[3]:.2f}")
    print("-" * 76)
    print(f"Option A latency (re-reads State every time) : {time_a*1000:6.0f} ms")
    print(f"Option B Request #1 (Cache Miss, hit={str(hit1):5s})  : {time_b1*1000:6.0f} ms")
    print(f"Option B Request #2 (Cache Hit!, hit={str(hit2):5s})  : {time_b2*1000:6.0f} ms  <-- {time_a/time_b2:.1f}x FASTER!")


if __name__ == "__main__":
    main()
