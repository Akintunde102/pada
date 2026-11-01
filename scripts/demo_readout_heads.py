"""
Educational Script: The 3 Readout Heads (Phase 3)
--------------------------------------------------
This script demonstrates how all 3 specialized judges work in practice:
1. Assemble a single prompt with 1 State and ALL 3 Question Types:
   - Question 1 (Binary / 'noul'): "Is this urgent?"
   - Question 2 (Choice / 'choice'): "Which team? Billing / Tech Support / Legal"
   - Question 3 (Score / 'score'): "Rate frustration from 1 to 5"
2. Run a single forward pass with Tree Attention masking.
3. Apply:
   - Judge 1: Noul Head (The Dimmer Switch -> Sigmoid)
   - Judge 2: Choice Head (The Spotlight -> Pointer Scorer + Softmax)
   - Judge 3: Score Head (The 5 Jars -> Distribution + Expected Value)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# =====================================================================
# The 3 Judges (Readout Heads) Defined as Simple PyTorch Modules
# =====================================================================

class NoulHead(nn.Module):
    """
    Judge 1: The Dimmer Switch (Binary Yes/No)
    Takes 1 thought vector at <|readout|>, outputs probability of True.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Linear layer compresses hidden_dim (896 numbers) into 1 single score (logit)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, readout_vector):
        # 1. Calculate single score
        score = self.proj(readout_vector)
        # 2. Squeeze score through sigmoid to get percentage (0.0 to 1.0)
        prob = torch.sigmoid(score)
        return prob.item()


class ChoiceHead(nn.Module):
    """
    Judge 2: The Spotlight (Pointer Scorer for Multiple Choice)
    Shines a comparison 'laser' from <|readout|> to each candidate <|opt|>.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # W_q transforms the question's readout thought (Query)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # W_k transforms each candidate option's thought (Key)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, readout_vector, option_vectors):
        # 1. Transform the query thought: [1, hidden_dim]
        q = self.W_q(readout_vector.unsqueeze(0))
        # 2. Transform the candidate options: [num_options, hidden_dim]
        k = self.W_k(option_vectors)

        # 3. Measure alignment (dot product) between query and every option
        # Formula: (q * k^T) / sqrt(d)
        scores = torch.matmul(q, k.T) / self.scale  # [1, num_options]

        # 4. Turn scores into percentages that add to 100% using softmax
        probabilities = F.softmax(scores, dim=-1).squeeze(0)
        return probabilities.tolist()


class ScoreHead(nn.Module):
    """
    Judge 3: The 5 Glass Jars (Ordinal Rating & Expected Value)
    Distributes marbles across K levels (e.g. 1 to 5) and finds the average balance.
    """
    def __init__(self, hidden_dim, num_levels=5):
        super().__init__()
        self.num_levels = num_levels
        # Linear layer projects 896 numbers into K bucket scores
        self.proj = nn.Linear(hidden_dim, num_levels)

    def forward(self, readout_vector):
        # 1. Get raw scores for all K jars
        scores = self.proj(readout_vector)
        # 2. Convert to probability distribution using softmax
        probs = F.softmax(scores, dim=-1)

        # 3. Calculate Expected Value: Sum(level * probability)
        # Levels are 1, 2, 3, 4, 5
        levels = torch.arange(1, self.num_levels + 1, dtype=scores.dtype, device=scores.device)
        expected_value = torch.sum(levels * probs).item()

        return probs.tolist(), expected_value


# =====================================================================
# Main Execution Pipeline
# =====================================================================

def main():
    print("=" * 65)
    print("Phase 3: The 3 Readout Heads in Action")
    print("=" * 65)

    # 1. Load Model and Tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\n[Step 1] Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Add our 7 road signs
    special_tokens = ["<|state|>", "<|q|>", "<|opt|>", "<|readout|>", "<|noul|>", "<|choice|>", "<|score|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    # 2. Define the Customer State and ALL 3 Question Types
    state_text = "<|state|> My payouts failed three times. Please fix this."
    q1_text = "<|q|> <|noul|> Is this message urgent? <|readout|>"
    q2_text = "<|q|> <|choice|> Which team should handle this? <|opt|> Billing <|opt|> Tech Support <|opt|> Legal <|readout|>"
    q3_text = "<|q|> <|score|> Rate customer frustration from 1 to 5: <|readout|>"

    # 3. Tokenize each piece
    state_ids = tokenizer.encode(state_text, add_special_tokens=False)
    q1_ids = tokenizer.encode(q1_text, add_special_tokens=False)
    q2_ids = tokenizer.encode(q2_text, add_special_tokens=False)
    q3_ids = tokenizer.encode(q3_text, add_special_tokens=False)

    all_ids = state_ids + q1_ids + q2_ids + q3_ids
    total_tokens = len(all_ids)
    print(f"  Total sequence length (State + 3 Questions): {total_tokens} tokens")

    # 4. Construct Segments (0 = State, 1 = Q1, 2 = Q2, 3 = Q3)
    segments = (
        [0] * len(state_ids) +
        [1] * len(q1_ids) +
        [2] * len(q2_ids) +
        [3] * len(q3_ids)
    )
    seg_tensor = torch.tensor(segments, dtype=torch.long)

    # 5. Tree Attention Mask: Causal & (State OR Same Question)
    row_idx = torch.arange(total_tokens).unsqueeze(1)
    col_idx = torch.arange(total_tokens).unsqueeze(0)
    causal = col_idx <= row_idx
    can_see_state = seg_tensor.unsqueeze(0) == 0
    same_branch = seg_tensor.unsqueeze(1) == seg_tensor.unsqueeze(0)
    tree_mask = causal & (can_see_state | same_branch)

    # 6. Restarting Position IDs (Order Invariance)
    S = len(state_ids)
    state_pos = list(range(S))
    q1_pos = [S + i for i in range(len(q1_ids))]
    q2_pos = [S + i for i in range(len(q2_ids))]
    q3_pos = [S + i for i in range(len(q3_ids))]
    pos_ids = torch.tensor([state_pos + q1_pos + q2_pos + q3_pos], dtype=torch.long)

    # 7. Convert mask to 4D additive format: 0.0 allowed, -inf blocked
    custom_4d_mask = torch.zeros((1, 1, total_tokens, total_tokens), dtype=torch.float32)
    custom_4d_mask.masked_fill_(~tree_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # 8. Run Single Prefill Pass
    print("\n[Step 2] Running single forward pass through 28 layers...")
    input_ids_tensor = torch.tensor([all_ids], dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_tensor,
            attention_mask=custom_4d_mask,
            position_ids=pos_ids
        )

    hidden_states = outputs.last_hidden_state  # [1, total_tokens, 896]
    print(f"  [OK] Single pass complete! Hidden states shape: {list(hidden_states.shape)}")

    # 9. Bookkeeping: Find the coordinates of our flags (<|readout|> and <|opt|>)
    readout_id = tokenizer.convert_tokens_to_ids("<|readout|>")
    opt_id = tokenizer.convert_tokens_to_ids("<|opt|>")

    # Readout positions for each question
    readout_indices = [idx for idx, t_id in enumerate(all_ids) if t_id == readout_id]
    q1_readout_idx = readout_indices[0]
    q2_readout_idx = readout_indices[1]
    q3_readout_idx = readout_indices[2]

    # Option positions for Q2
    q2_start = len(state_ids) + len(q1_ids)
    q2_end = q2_start + len(q2_ids)
    q2_opt_indices = [idx for idx in range(q2_start, q2_end) if all_ids[idx] == opt_id]

    # =================================================================
    # Step 10: Call the 3 Judges!
    # =================================================================
    hidden_dim = hidden_states.shape[-1]
    model_dtype = hidden_states.dtype
    noul_judge = NoulHead(hidden_dim).to(dtype=model_dtype)
    choice_judge = ChoiceHead(hidden_dim).to(dtype=model_dtype)
    score_judge = ScoreHead(hidden_dim, num_levels=5).to(dtype=model_dtype)

    print("\n" + "=" * 65)
    print("RESULTS DELIVERED BY THE 3 READOUT HEADS:")
    print("=" * 65)

    # Judge 1: The Dimmer Switch
    q1_vector = hidden_states[0, q1_readout_idx, :]
    p_urgent = noul_judge(q1_vector)
    print(f"\n[Judge 1: The Dimmer Switch (noul)]")
    print(f"  Question:  'Is this message urgent?'")
    print(f"  P(Urgent): {p_urgent:.4f} ({p_urgent * 100:.1f}%)")

    # Judge 2: The Spotlight
    q2_readout_vector = hidden_states[0, q2_readout_idx, :]
    q2_option_vectors = hidden_states[0, q2_opt_indices, :]  # [3, 896]
    choice_probs = choice_judge(q2_readout_vector, q2_option_vectors)
    options = ["Billing", "Tech Support", "Legal"]

    print(f"\n[Judge 2: The Spotlight (choice)]")
    print(f"  Question:  'Which team should handle this?'")
    for opt_name, prob in zip(options, choice_probs):
        bar = "#" * int(prob * 30)
        print(f"  - {opt_name:<14}: {prob:.4f} ({prob * 100:5.1f}%)  {bar}")

    # Judge 3: The 5 Jars
    q3_vector = hidden_states[0, q3_readout_idx, :]
    distribution, expected_score = score_judge(q3_vector)
    print(f"\n[Judge 3: The 5 Glass Jars (score)]")
    print(f"  Question:  'Rate customer frustration from 1 to 5'")
    for level, prob in enumerate(distribution, 1):
        print(f"  - Level {level}: {prob * 100:5.1f}%")
    print(f"  >> Expected Value (Rating): {expected_score:.2f} / 5.00")

    print("\n" + "=" * 65)
    print("Phase 3 Complete: All 3 readout heads fully operational!")
    print("=" * 65)


if __name__ == "__main__":
    main()
