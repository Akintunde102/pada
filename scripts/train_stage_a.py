"""
Educational Script: Phase 5, Stage A -- Training the 3 Judges
--------------------------------------------------------------
Goal: take the SAME pipeline as demo_multi_scenarios.py, but this time TEACH the
3 judges (readout heads) using a small set of labelled examples.

The plan, in plain English:
  1. Load Qwen and FREEZE it (its 24 layers never change in this script).
  2. Run every training example through Qwen ONCE and save the thought vectors
     at the <|readout|> and <|opt|> positions. (Because Qwen is frozen, these
     vectors never change, so we only need to compute them once. Fast!)
  3. Test the UNTRAINED judges on held-out scenarios they will never train on.
  4. Train the judges with proper scoring rules (cross-entropy + Brier).
  5. Test the SAME held-out scenarios again and compare BEFORE vs AFTER.

Honest caveats:
  - We only have 28 training examples. Real Jev training uses millions.
  - We do NOT train the 7 special-token embeddings here (that would need
    gradients flowing through the frozen Qwen, which is slow on a CPU).
  - So expect "clearly better", not "perfect".
"""

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Fixed seeds -> the "random" starting weights are the same on every run.
torch.manual_seed(0)
random.seed(0)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
SPECIAL_TOKENS = ["<|state|>", "<|q|>", "<|opt|>", "<|readout|>",
                  "<|noul|>", "<|choice|>", "<|score|>"]


# =====================================================================
# 1. THE QUESTIONS (one fixed set of 3 questions per area of life)
# =====================================================================
DOMAINS = {
    "support": {
        "noul": "Is this message urgent?",
        "choice": "Which department should handle this?",
        "options": ["Billing", "Tech Support", "Legal"],
        "score": "Rate customer frustration from 1 to 5:",
    },
    "medical": {
        "noul": "Does this patient require immediate emergency intervention?",
        "choice": "Which medical unit should receive this patient?",
        "options": ["Cardiology / ER", "Dermatology", "Routine Checkup", "Dental"],
        "score": "Rate clinical severity from 1 to 5:",
    },
    "fraud": {
        "noul": "Should we block this transaction for fraud?",
        "choice": "What is the primary risk category?",
        "options": ["Payment Fraud", "Friendly Return", "Shipping Delay"],
        "score": "Rate fraud risk level from 1 to 5:",
    },
    "restaurant": {
        "noul": "Is this customer asking for a refund?",
        "choice": "What is the main topic of this review?",
        "options": ["Food & Service Praise", "Health Code Violation", "Parking Complaint"],
        "score": "Rate overall guest satisfaction from 1 to 5:",
    },
}


# =====================================================================
# 2. THE TRAINING DATA (Phase 4 in miniature)
# Each row: (area, state, p_true target, correct option, 5-jar target)
# Note the SOFT targets: 0.3 means "borderline, 3 in 10 people would say yes".
# =====================================================================
TRAIN = [
    # --- Customer support ---
    ("support", "I was charged twice for my subscription this month, please refund one.", 0.30, "Billing", [0, .2, .5, .3, 0]),
    ("support", "The app crashes every time I open the settings page.", 0.30, "Tech Support", [0, .3, .5, .2, 0]),
    ("support", "Your company is using my photos without permission. My lawyer will contact you.", 0.80, "Legal", [0, 0, .1, .4, .5]),
    ("support", "Just wondering how to change my profile picture, no rush.", 0.05, "Tech Support", [.7, .3, 0, 0, 0]),
    ("support", "My salary deposit never arrived and rent is due tomorrow! Help!", 0.95, "Billing", [0, 0, 0, .3, .7]),
    ("support", "I received a court subpoena regarding my account data.", 0.70, "Legal", [0, .2, .5, .3, 0]),
    ("support", "I can't log in, the password reset email never comes.", 0.60, "Tech Support", [0, .1, .4, .4, .1]),
    # --- Medical triage ---
    ("medical", "Patient has a mild itchy rash on the forearm for two weeks.", 0.05, "Dermatology", [.6, .4, 0, 0, 0]),
    ("medical", "Elderly man collapsed, irregular heartbeat, struggling to breathe.", 0.98, "Cardiology / ER", [0, 0, 0, .2, .8]),
    ("medical", "Patient requests an annual physical and feels healthy.", 0.01, "Routine Checkup", [.95, .05, 0, 0, 0]),
    ("medical", "Severe toothache and swollen gum for three days.", 0.20, "Dental", [0, .3, .6, .1, 0]),
    ("medical", "Woman reports tight chest pressure and dizziness while climbing stairs.", 0.85, "Cardiology / ER", [0, 0, .1, .5, .4]),
    ("medical", "Teenager with an acne flare-up wants treatment options.", 0.02, "Dermatology", [.8, .2, 0, 0, 0]),
    ("medical", "Child chipped a front tooth playing football, no bleeding.", 0.10, "Dental", [.4, .5, .1, 0, 0]),
    # --- Fraud screening ---
    ("fraud", "Loyal customer of six years asks where her shoe order is, tracking stuck for a week.", 0.02, "Shipping Delay", [.9, .1, 0, 0, 0]),
    ("fraud", "Twenty failed card attempts in two minutes with different card numbers, then one success.", 0.97, "Payment Fraud", [0, 0, 0, .1, .9]),
    ("fraud", "Customer says the jacket didn't fit and wants to send it back.", 0.03, "Friendly Return", [.8, .2, 0, 0, 0]),
    ("fraud", "New account, billing and shipping in different countries, expensive electronics, rush delivery.", 0.80, "Payment Fraud", [0, 0, .1, .4, .5]),
    ("fraud", "Order is held at customs and the customer asks for an update.", 0.02, "Shipping Delay", [.9, .1, 0, 0, 0]),
    ("fraud", "Customer changed her mind about a lamp and requests a return label.", 0.02, "Friendly Return", [.9, .1, 0, 0, 0]),
    ("fraud", "A card the bank reported stolen was just used to buy three expensive phones.", 0.99, "Payment Fraud", [0, 0, 0, 0, 1]),
    # --- Restaurant reviews ---
    ("restaurant", "Found a cockroach in my soup and the kitchen smelled awful. I want my money back.", 0.95, "Health Code Violation", [.9, .1, 0, 0, 0]),
    ("restaurant", "Lovely dinner, the steak was perfect and the staff were friendly.", 0.01, "Food & Service Praise", [0, 0, 0, .2, .8]),
    ("restaurant", "Food was fine but there was nowhere to park, we circled for 40 minutes.", 0.10, "Parking Complaint", [0, .3, .5, .2, 0]),
    ("restaurant", "Raw chicken was served and my husband got food poisoning. Refund us now.", 0.98, "Health Code Violation", [1, 0, 0, 0, 0]),
    ("restaurant", "Best tiramisu in town and the chef came out to say hello!", 0.00, "Food & Service Praise", [0, 0, 0, .1, .9]),
    ("restaurant", "The valet lost our car keys and the lot was full. Very annoying.", 0.40, "Parking Complaint", [.3, .5, .2, 0, 0]),
    ("restaurant", "Our waitress was sweet and the pizza came out fast and hot.", 0.00, "Food & Service Praise", [0, 0, .05, .35, .6]),
]

# =====================================================================
# 3. HELD-OUT TEST SCENARIOS (the judges NEVER train on these)
# =====================================================================
TEST = [
    ("support", "My payouts failed three times. Please fix this."),
    ("support", "Hi, how do I download my invoice from last month? Thanks."),
    ("medical", "45-year-old patient reporting sudden crushing chest pain radiating to the left arm and sweating."),
    ("medical", "Patient has a small mole that changed color slightly, no other symptoms."),
    ("fraud", "Account created 90 seconds ago via VPN, ordering six $500 digital gift cards with a foreign prepaid card."),
    ("fraud", "Customer wants to return unopened headphones within 30 days."),
    ("restaurant", "The homemade pasta was incredible, our waiter Marco was so warm and attentive, and we loved the patio!"),
    ("restaurant", "There were mouse droppings near our table and the bathroom was filthy."),
]


# =====================================================================
# 4. PACKING + ONE FORWARD PASS (Phase 2, same as before)
# =====================================================================
def extract_vectors(model, tokenizer, area, state, options):
    """Pack State + 3 questions, run ONE frozen forward pass, and return
    only the thought vectors the judges need."""
    q = DOMAINS[area]
    parts = [
        f"<|state|> {state}",
        f"<|q|> <|noul|> {q['noul']} <|readout|>",
        f"<|q|> <|choice|> {q['choice']} " + " ".join(f"{o} <|opt|>" for o in options) + " <|readout|>",
        f"<|q|> <|score|> {q['score']} <|readout|>",
    ]
    pieces = [tokenizer.encode(p, add_special_tokens=False) for p in parts]

    ids, seg, pos = [], [], []
    S = len(pieces[0])                       # length of the State
    for k, piece in enumerate(pieces):
        ids += piece
        seg += [k] * len(piece)              # 0 = State, 1/2/3 = questions
        start = 0 if k == 0 else S           # every question restarts its ruler at S
        pos += list(range(start, start + len(piece)))

    n = len(ids)
    seg_t = torch.tensor(seg)
    rows, cols = torch.arange(n).unsqueeze(1), torch.arange(n).unsqueeze(0)
    allowed = (cols <= rows) & ((seg_t.unsqueeze(0) == 0) | (seg_t.unsqueeze(1) == seg_t.unsqueeze(0)))
    mask = torch.zeros((1, 1, n, n)).masked_fill_(~allowed, float("-inf"))

    with torch.no_grad():                    # frozen: no training math needed
        h = model(input_ids=torch.tensor([ids]), attention_mask=mask,
                  position_ids=torch.tensor([pos])).last_hidden_state[0].float()

    readout_id = tokenizer.convert_tokens_to_ids("<|readout|>")
    opt_id = tokenizer.convert_tokens_to_ids("<|opt|>")
    readouts = [i for i, t in enumerate(ids) if t == readout_id]
    opts = [i for i, t in enumerate(ids) if t == opt_id]

    return {
        "noul": h[readouts[0]],              # Q1 finish line
        "choice_q": h[readouts[1]],          # Q2 finish line
        "choice_opts": h[opts],              # one vector per <|opt|>
        "score": h[readouts[2]],             # Q3 finish line
    }


# =====================================================================
# 5. THE 3 JUDGES (now returning raw scores, a.k.a. "logits")
# LayerNorm = a volume knob that rescales the 896 numbers to a steady range,
# which makes training much more stable.
# =====================================================================
class NoulHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)

    def forward(self, h):
        return self.proj(self.norm(h)).squeeze(-1)          # one score


class ChoiceHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        # Per jev-rebuild-plan.md (Phase 3): initialize W_q and W_k near identity
        # so they start in Qwen's native representation space instead of scrambling it.
        with torch.no_grad():
            self.W_q.weight.copy_(torch.eye(d) + 0.01 * torch.randn(d, d))
            self.W_k.weight.copy_(torch.eye(d) + 0.01 * torch.randn(d, d))
        self.scale = math.sqrt(d)

    def forward(self, h_readout, h_opts):
        q = self.W_q(self.norm(h_readout))                  # [d]
        k = self.W_k(self.norm(h_opts))                     # [num_opts, d]
        return (k @ q) / self.scale                         # one score per option


class ScoreHead(nn.Module):
    def __init__(self, d, levels=5):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, levels)

    def forward(self, h):
        return self.proj(self.norm(h))                      # one score per jar


# =====================================================================
# 6. THE PENALTY (proper scoring rules)
# =====================================================================
def penalty(heads, vecs, target):
    # Judge 1: binary cross-entropy against the soft yes/no target
    l_noul = F.binary_cross_entropy_with_logits(
        heads["noul"](vecs["noul"]), torch.tensor(target["p_true"]))

    # Judge 2: cross-entropy + Brier against the correct option
    c_logits = heads["choice"](vecs["choice_q"], vecs["choice_opts"])
    c_target = target["choice"]
    l_choice = (-(c_target * F.log_softmax(c_logits, -1)).sum()
                + ((F.softmax(c_logits, -1) - c_target) ** 2).sum())

    # Judge 3: cross-entropy + Brier against the 5-jar target
    s_logits = heads["score"](vecs["score"])
    s_target = target["score"]
    l_score = (-(s_target * F.log_softmax(s_logits, -1)).sum()
               + ((F.softmax(s_logits, -1) - s_target) ** 2).sum())

    return l_noul + l_choice + l_score


def predict(heads, vecs, options):
    with torch.no_grad():
        p = torch.sigmoid(heads["noul"](vecs["noul"])).item()
        c = F.softmax(heads["choice"](vecs["choice_q"], vecs["choice_opts"]), -1)
        s = F.softmax(heads["score"](vecs["score"]), -1)
        expected = (torch.arange(1, 6).float() * s).sum().item()
    best = int(c.argmax())
    return p, options[best], c[best].item(), expected


# =====================================================================
# 7. MAIN
# =====================================================================
def main():
    print("[1] Loading and freezing Qwen (in float32 for fast CPU execution)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model.resize_token_embeddings(len(tokenizer))

    # Warm-start the 7 new special tokens from real words Qwen already understands,
    # instead of leaving them as meaningless "average gray mush" vectors!
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

    model.eval()
    for param in model.parameters():
        param.requires_grad = False          # FREEZE: Qwen's weights never change

    # --- Pre-compute thought vectors for training (with option shuffling) ---
    print("[2] Running 28 training examples through frozen Qwen...", flush=True)
    t0 = time.time()
    train_items = []
    for idx, (area, state, p_true, correct, jars) in enumerate(TRAIN, 1):
        options = DOMAINS[area]["options"][:]
        random.shuffle(options)              # shuffle option order so position can't be memorized
        vecs = extract_vectors(model, tokenizer, area, state, options)
        target = {
            "p_true": p_true,
            "choice": torch.tensor([1.0 if o == correct else 0.0 for o in options]),
            "score": torch.tensor(jars, dtype=torch.float32),
        }
        train_items.append((vecs, target))
        if idx % 7 == 0:
            print(f"    encoded {idx}/{len(TRAIN)} training examples ({time.time() - t0:.1f}s)", flush=True)

    print("[3] Running 8 held-out test scenarios...", flush=True)
    test_items = [(area, state, extract_vectors(model, tokenizer, area, state, DOMAINS[area]["options"]))
                  for area, state in TEST]

    d = model.config.hidden_size
    heads = nn.ModuleDict({"noul": NoulHead(d), "choice": ChoiceHead(d), "score": ScoreHead(d)})

    before = [predict(heads, v, DOMAINS[a]["options"]) for a, _, v in test_items]

    # --- Training loop ---
    print("[4] Training the 3 judges (Stage A)...", flush=True)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=0.01)
    for epoch in range(1, 201):
        random.shuffle(train_items)
        total = 0.0
        for vecs, target in train_items:
            # Also permute option vectors at every step so the Spotlight never memorizes slot index
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
                "score": target["score"],
            }
            loss = penalty(heads, step_vecs, step_target)  # 1. how wrong were we?
            optimizer.zero_grad()                          # 2. clear old nudges
            loss.backward()                                # 3. work out which way to nudge each weight
            optimizer.step()                               # 4. apply the nudges
            total += loss.item()
        if epoch in (1, 10, 50, 100, 200):
            print(f"    epoch {epoch:3d}  average penalty = {total / len(train_items):.3f}", flush=True)

    after = [predict(heads, v, DOMAINS[a]["options"]) for a, _, v in test_items]

    # --- Before vs After ---
    print("\n" + "=" * 78)
    print("HELD-OUT RESULTS (never seen in training): BEFORE -> AFTER")
    print("=" * 78)
    for (area, state, _), b, a in zip(test_items, before, after):
        q = DOMAINS[area]
        print(f"\n[{area}] \"{state}\"")
        print(f"  {q['noul']:<62} {b[0]*100:5.1f}% -> {a[0]*100:5.1f}%")
        print(f"  {q['choice']:<62}")
        print(f"      before: {b[1]} ({b[2]*100:.0f}%)   after: {a[1]} ({a[2]*100:.0f}%)")
        print(f"  {q['score']:<62} {b[3]:4.2f} -> {a[3]:4.2f}")


if __name__ == "__main__":
    main()
