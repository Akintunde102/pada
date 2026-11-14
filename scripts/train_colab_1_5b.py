"""
Stage A & Stage B Training Script for Qwen2.5-1.5B + LoRA + Readout Heads
-------------------------------------------------------------------------
Trains:
  - Stage A: NoulHead, ChoiceHead (Pointer Scorer), and UniversalScoreHead on frozen Qwen2.5-1.5B vectors
  - Stage B: Joint fine-tuning of rank-8 LoRA (q_proj, v_proj) + Readout Heads with BCE + Brier loss
Saves:
  - checkpoints/jev_1_5b_lora_adapter
  - checkpoints/jev_1_5b_heads_and_T.pt
"""

import math
import os
import random
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(42)
random.seed(42)

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
SPECIAL_TOKENS = ["<|state|>", "<|q|>", "<|opt|>", "<|readout|>",
                  "<|noul|>", "<|choice|>", "<|score|>"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Running on device: {device.upper()}")

# Tokenizer & Backbone setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
model.resize_token_embeddings(len(tokenizer))

# Warm-start special tokens from semantic anchors
warm_words = {
    "<|state|>": "Context",
    "<|q|>": "Question",
    "<|opt|>": ":",
    "<|readout|>": "Answer",
    "<|noul|>": "Yes",
    "<|choice|>": "Option",
    "<|score|>": "Rating",
}
with torch.no_grad():
    embed_weight = model.get_input_embeddings().weight
    for spec_tok, warm_word in warm_words.items():
        spec_id = tokenizer.convert_tokens_to_ids(spec_tok)
        warm_id = tokenizer.encode(warm_word, add_special_tokens=False)[0]
        embed_weight[spec_id].copy_(embed_weight[warm_id])

# Attach LoRA (rank 8, alpha 16 on q_proj & v_proj)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION",
)
lora_model = get_peft_model(model, lora_config)
d = model.config.hidden_size  # 1536


# Readout Heads
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
        with torch.no_grad():
            self.W_q.weight.copy_(torch.eye(d) + 0.01 * torch.randn(d, d))
            self.W_k.weight.copy_(torch.eye(d) + 0.01 * torch.randn(d, d))
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


heads = nn.ModuleDict({
    "noul": NoulHead(d),
    "choice": ChoiceHead(d),
    "score": UniversalScoreHead(d),
})

DOMAINS = {
    "review": {
        "noul": "Is this customer review positive?",
        "choice": "What is the overall sentiment tier of this review?",
        "options": ["Negative Review", "Neutral Review", "Positive Review"],
        "score": "Rate overall satisfaction from 0 to 1:",
    }
}


def extract_vectors(net, tok, area, state, options):
    dev = next(net.parameters()).device
    q = DOMAINS[area]
    parts = [
        f"<|state|> {state}",
        f"<|q|> <|noul|> {q['noul']} <|readout|>",
        f"<|q|> <|choice|> {q['choice']} " + " ".join(f"{o} <|opt|>" for o in options) + " <|readout|>",
        f"<|q|> <|score|> {q['score']} <|readout|>",
    ]
    pieces = [tok.encode(p, add_special_tokens=False) for p in parts]

    ids, seg, pos = [], [], []
    S = len(pieces[0])
    for k, piece in enumerate(pieces):
        ids += piece
        seg += [k] * len(piece)
        start = 0 if k == 0 else S
        pos += list(range(start, start + len(piece)))

    n = len(ids)
    seg_t = torch.tensor(seg, device=dev)
    rows = torch.arange(n, device=dev).unsqueeze(1)
    cols = torch.arange(n, device=dev).unsqueeze(0)
    allowed = (cols <= rows) & ((seg_t.unsqueeze(0) == 0) | (seg_t.unsqueeze(1) == seg_t.unsqueeze(0)))
    mask = torch.zeros((1, 1, n, n), device=dev).masked_fill_(~allowed, float("-inf"))

    with torch.no_grad():
        h = net(
            input_ids=torch.tensor([ids], device=dev),
            attention_mask=mask,
            position_ids=torch.tensor([pos], device=dev),
        ).last_hidden_state[0].float().cpu()

    readout_id = tok.convert_tokens_to_ids("<|readout|>")
    opt_id = tok.convert_tokens_to_ids("<|opt|>")
    readouts = [i for i, t in enumerate(ids) if t == readout_id]
    opts = [i for i, t in enumerate(ids) if t == opt_id]

    return {
        "noul": h[readouts[0]],
        "choice_q": h[readouts[1]],
        "choice_opts": h[opts],
        "score": h[readouts[2]],
    }


def penalty(heads_dict, vecs, target):
    l_noul = F.binary_cross_entropy_with_logits(
        heads_dict["noul"](vecs["noul"]), torch.tensor(target["p_true"]))

    c_logits = heads_dict["choice"](vecs["choice_q"], vecs["choice_opts"])
    c_target = target["choice"]
    l_choice = (-(c_target * F.log_softmax(c_logits, -1)).sum()
                + ((F.softmax(c_logits, -1) - c_target) ** 2).sum())

    s_raw = heads_dict["score"](vecs["score"])
    s_prob = torch.sigmoid(s_raw)
    s_target = torch.tensor(target["norm_score"], dtype=torch.float32)
    l_score = F.binary_cross_entropy_with_logits(s_raw, s_target) + ((s_prob - s_target) ** 2)

    return l_noul + l_choice + l_score


def main():
    ds = load_dataset("SetFit/amazon_reviews_multi_en", split="train").shuffle(seed=42).select(range(1000))

    train_items = []
    for item in ds:
        text = item["text"][:250].strip()
        label = item["label"]
        norm_score = 0.05 + (label / 4.0) * 0.90
        if label >= 3:
            p_true, tier = 0.95, "Positive Review"
        elif label <= 1:
            p_true, tier = 0.05, "Negative Review"
        else:
            p_true, tier = 0.50, "Neutral Review"
        train_items.append(("review", text, p_true, tier, norm_score))

    # Stage A: Warm up heads on frozen Qwen vectors
    cached_train = []
    for area, state, p_true, correct, norm_s in train_items:
        options = DOMAINS[area]["options"][:]
        random.shuffle(options)
        vecs = extract_vectors(lora_model, tokenizer, area, state, options)
        cached_train.append((vecs, {
            "p_true": p_true,
            "choice": torch.tensor([1.0 if o == correct else 0.0 for o in options]),
            "norm_score": norm_s
        }))

    opt_heads = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=0.01)
    for _ in range(1, 21):
        random.shuffle(cached_train)
        for vecs, target in cached_train:
            perm = torch.randperm(vecs["choice_opts"].shape[0])
            step_vecs = {
                "noul": vecs["noul"],
                "choice_q": vecs["choice_q"],
                "choice_opts": vecs["choice_opts"][perm],
                "score": vecs["score"],
            }
            step_target = {
                "p_true": target["p_true"],
                "choice": target["choice"][perm],
                "norm_score": target["norm_score"],
            }
            loss = penalty(heads, step_vecs, step_target)
            opt_heads.zero_grad()
            loss.backward()
            opt_heads.step()

    # Stage B: Train LoRA + Heads together
    all_params = [p for p in lora_model.parameters() if p.requires_grad] + list(heads.parameters())
    opt_b = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0.01)
    lora_model.train()
    heads.train()

    readout_id = tokenizer.convert_tokens_to_ids("<|readout|>")
    opt_id = tokenizer.convert_tokens_to_ids("<|opt|>")
    stage_b_slice = list(train_items[:300])

    for _ in range(1, 3):
        random.shuffle(stage_b_slice)
        for area, state, p_true, correct, norm_s in stage_b_slice:
            shuffled_opts = DOMAINS[area]["options"][:]
            random.shuffle(shuffled_opts)
            q = DOMAINS[area]
            parts = [
                f"<|state|> {state}",
                f"<|q|> <|noul|> {q['noul']} <|readout|>",
                f"<|q|> <|choice|> {q['choice']} " + " ".join(f"{o} <|opt|>" for o in shuffled_opts) + " <|readout|>",
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
            seg_t = torch.tensor(seg, device=device)
            rows = torch.arange(n, device=device).unsqueeze(1)
            cols = torch.arange(n, device=device).unsqueeze(0)
            allowed = (cols <= rows) & ((seg_t.unsqueeze(0) == 0) | (seg_t.unsqueeze(1) == seg_t.unsqueeze(0)))
            mask = torch.zeros((1, 1, n, n), device=device).masked_fill_(~allowed, float("-inf"))

            out = lora_model(
                input_ids=torch.tensor([ids], device=device),
                attention_mask=mask,
                position_ids=torch.tensor([pos], device=device),
            ).last_hidden_state[0].float().cpu()

            readouts = [i for i, t in enumerate(ids) if t == readout_id]
            opts = [i for i, t in enumerate(ids) if t == opt_id]

            step_vecs = {
                "noul": out[readouts[0]],
                "choice_q": out[readouts[1]],
                "choice_opts": out[opts],
                "score": out[readouts[2]],
            }
            step_target = {
                "p_true": p_true,
                "choice": torch.tensor([1.0 if o == correct else 0.0 for o in shuffled_opts]),
                "norm_score": norm_s,
            }

            loss = penalty(heads, step_vecs, step_target)
            opt_b.zero_grad()
            loss.backward()
            opt_b.step()

    lora_model.eval()
    heads.eval()

    os.makedirs("checkpoints", exist_ok=True)
    lora_model.save_pretrained("checkpoints/jev_1_5b_lora_adapter")
    torch.save({
        "heads": heads.state_dict(),
        "T": {"noul": 1.5, "choice": 2.0, "score": 1.5},
        "model_name": MODEL_NAME,
        "hidden_size": d,
    }, "checkpoints/jev_1_5b_heads_and_T.pt")
    print("[+] Saved trained LoRA adapter and heads to checkpoints/!")

    try:
        from google.colab import files
        subprocess.run(["zip", "-rq", "jev_1_5b_checkpoint.zip", "checkpoints/jev_1_5b_lora_adapter", "checkpoints/jev_1_5b_heads_and_T.pt"])
        files.download("jev_1_5b_checkpoint.zip")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
