"""
Educational Script: Testing Jev Across Diverse Real-Life Scenarios
-------------------------------------------------------------------
We run 4 completely different areas of life through our Phase 1-3 pipeline
(before Phase 5 training) so we can see how the untrained judges behave:
1. Customer Support (Failed payouts)
2. Medical Triage (Chest pain & shortness of breath)
3. Fraud Screening (Suspicious gift card purchase)
4. Restaurant Review (Happy customer dining experience)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# Set a fixed random seed so the untrained weights are reproducible
torch.manual_seed(42)


class NoulHead(nn.Module):
    """Judge 1: The Dimmer Switch (Yes/No)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, readout_vector):
        return torch.sigmoid(self.proj(readout_vector)).item()


class ChoiceHead(nn.Module):
    """Judge 2: The Spotlight (Multiple Choice Pointer Scorer)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, readout_vector, option_vectors):
        q = self.W_q(readout_vector.unsqueeze(0))
        k = self.W_k(option_vectors)
        scores = torch.matmul(q, k.T) / self.scale
        return F.softmax(scores, dim=-1).squeeze(0).tolist()


class ScoreHead(nn.Module):
    """Judge 3: The 5 Glass Jars (1 to 5 Rating)"""
    def __init__(self, hidden_dim, num_levels=5):
        super().__init__()
        self.num_levels = num_levels
        self.proj = nn.Linear(hidden_dim, num_levels)

    def forward(self, readout_vector):
        scores = self.proj(readout_vector)
        probs = F.softmax(scores, dim=-1)
        levels = torch.arange(1, self.num_levels + 1, dtype=scores.dtype, device=scores.device)
        expected = torch.sum(levels * probs).item()
        return probs.tolist(), expected


def evaluate_scenario(model, tokenizer, noul_judge, choice_judge, score_judge, scenario):
    """Packs 1 State + 3 Questions with Tree Attention and runs 1 forward pass."""
    state_text = f"<|state|> {scenario['state']}"
    q1_text = f"<|q|> <|noul|> {scenario['q_noul']} <|readout|>"

    opts_str = " ".join([f"<|opt|> {opt}" for opt in scenario["q_choice_opts"]])
    q2_text = f"<|q|> <|choice|> {scenario['q_choice']} {opts_str} <|readout|>"

    q3_text = f"<|q|> <|score|> {scenario['q_score']} <|readout|>"

    state_ids = tokenizer.encode(state_text, add_special_tokens=False)
    q1_ids = tokenizer.encode(q1_text, add_special_tokens=False)
    q2_ids = tokenizer.encode(q2_text, add_special_tokens=False)
    q3_ids = tokenizer.encode(q3_text, add_special_tokens=False)

    all_ids = state_ids + q1_ids + q2_ids + q3_ids
    total_tokens = len(all_ids)

    segments = [0] * len(state_ids) + [1] * len(q1_ids) + [2] * len(q2_ids) + [3] * len(q3_ids)
    seg_tensor = torch.tensor(segments, dtype=torch.long)

    row_idx = torch.arange(total_tokens).unsqueeze(1)
    col_idx = torch.arange(total_tokens).unsqueeze(0)
    tree_mask = (col_idx <= row_idx) & ((seg_tensor.unsqueeze(0) == 0) | (seg_tensor.unsqueeze(1) == seg_tensor.unsqueeze(0)))

    S = len(state_ids)
    pos_ids = torch.tensor([[
        *range(S),
        *[S + i for i in range(len(q1_ids))],
        *[S + i for i in range(len(q2_ids))],
        *[S + i for i in range(len(q3_ids))],
    ]], dtype=torch.long)

    custom_4d_mask = torch.zeros((1, 1, total_tokens, total_tokens), dtype=torch.float32)
    custom_4d_mask.masked_fill_(~tree_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    with torch.no_grad():
        outputs = model(
            input_ids=torch.tensor([all_ids], dtype=torch.long),
            attention_mask=custom_4d_mask,
            position_ids=pos_ids,
        )

    h = outputs.last_hidden_state
    readout_id = tokenizer.convert_tokens_to_ids("<|readout|>")
    opt_id = tokenizer.convert_tokens_to_ids("<|opt|>")

    readout_indices = [idx for idx, t_id in enumerate(all_ids) if t_id == readout_id]
    q2_start = len(state_ids) + len(q1_ids)
    q2_end = q2_start + len(q2_ids)
    q2_opt_indices = [idx for idx in range(q2_start, q2_end) if all_ids[idx] == opt_id]

    p_noul = noul_judge(h[0, readout_indices[0], :])
    p_choice = choice_judge(h[0, readout_indices[1], :], h[0, q2_opt_indices, :])
    dist_score, exp_score = score_judge(h[0, readout_indices[2], :])

    print("\n" + "=" * 68)
    print(f"SCENARIO: {scenario['title']}")
    print(f"STATE:    \"{scenario['state']}\"")
    print("-" * 68)
    print(f"  [noul]   {scenario['q_noul']}")
    print(f"           -> P(True): {p_noul:.4f} ({p_noul * 100:.1f}%)")
    print(f"  [choice] {scenario['q_choice']}")
    for opt, prob in zip(scenario["q_choice_opts"], p_choice):
        print(f"           - {opt:<18}: {prob * 100:5.1f}%")
    print(f"  [score]  {scenario['q_score']}")
    dist_str = " | ".join([f"L{i+1}:{p*100:4.1f}%" for i, p in enumerate(dist_score)])
    print(f"           -> [{dist_str}]  Expected: {exp_score:.2f} / 5.00")


def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    special_tokens = ["<|state|>", "<|q|>", "<|opt|>", "<|readout|>", "<|noul|>", "<|choice|>", "<|score|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    hidden_dim = model.config.hidden_size
    model_dtype = next(model.parameters()).dtype

    noul_judge = NoulHead(hidden_dim).to(dtype=model_dtype)
    choice_judge = ChoiceHead(hidden_dim).to(dtype=model_dtype)
    score_judge = ScoreHead(hidden_dim, num_levels=5).to(dtype=model_dtype)

    scenarios = [
        {
            "title": "1. FinTech Support (Failed Payouts)",
            "state": "My payouts failed three times. Please fix this.",
            "q_noul": "Is this message urgent?",
            "q_choice": "Which department should handle this?",
            "q_choice_opts": ["Billing", "Tech Support", "Legal"],
            "q_score": "Rate customer frustration from 1 to 5:",
        },
        {
            "title": "2. Healthcare / ER Triage (Chest Pain)",
            "state": "45-year-old patient reporting sudden crushing chest pain radiating to the left arm and sweating.",
            "q_noul": "Does this patient require immediate emergency intervention?",
            "q_choice": "Which medical unit should receive this patient?",
            "q_choice_opts": ["Cardiology / ER", "Dermatology", "Routine Checkup", "Dental"],
            "q_score": "Rate clinical severity from 1 to 5:",
        },
        {
            "title": "3. E-Commerce Security (Suspicious Order)",
            "state": "Account created 90 seconds ago via VPN, ordering six $500 digital gift cards with a foreign prepaid card.",
            "q_noul": "Should we block this transaction for fraud?",
            "q_choice": "What is the primary risk category?",
            "q_choice_opts": ["Payment Fraud", "Friendly Return", "Shipping Delay"],
            "q_score": "Rate fraud risk level from 1 to 5:",
        },
        {
            "title": "4. Hospitality / Dining Review (Happy Guest)",
            "state": "The homemade pasta was incredible, our waiter Marco was so warm and attentive, and we loved the patio!",
            "q_noul": "Is this customer asking for a refund?",
            "q_choice": "What is the main topic of this review?",
            "q_choice_opts": ["Food & Service Praise", "Health Code Violation", "Parking Complaint"],
            "q_score": "Rate overall guest satisfaction from 1 to 5:",
        },
    ]

    for sc in scenarios:
        evaluate_scenario(model, tokenizer, noul_judge, choice_judge, score_judge, sc)


if __name__ == "__main__":
    main()
