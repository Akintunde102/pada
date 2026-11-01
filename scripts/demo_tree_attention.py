"""
Educational Script: Tree Attention & Special Tokens (Phase 2)
--------------------------------------------------------------
This script demonstrates the core mechanisms of Phase 2:
1. Adding our 7 custom special road signs to the tokenizer and resizing the model.
2. Packing a shared State and two separate Questions into one single sequence.
3. Creating Segment IDs (0 for State, 1 for Q1, 2 for Q2).
4. Generating the Tree Attention Mask (the digital blindfolds).
5. Generating Position IDs that restart at the State length.
6. Printing the visual 2D attention grid.
7. Running the model in a single forward pass and extracting each question's <|readout|>.
"""

import torch
from transformers import AutoTokenizer, AutoModel


def main():
    print("=" * 65)
    print("Phase 2: Tree Attention, Special Tokens & Order Invariance")
    print("=" * 65)

    # -------------------------------------------------------------
    # Step 1: Load the Backbone and Add Special Road Signs
    # -------------------------------------------------------------
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\n[Step 1] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # The 7 special road signs required by Jev
    special_tokens = [
        "<|state|>",    # Marks customer context
        "<|q|>",        # Marks question start
        "<|opt|>",      # Marks an option inside a choice question
        "<|readout|>",  # Marks where the decision is measured
        "<|noul|>",     # Binary (Yes/No) type
        "<|choice|>",   # Multi-choice type
        "<|score|>",    # 1-to-5 rating type
    ]

    print("  Adding 7 special tokens to tokenizer...")
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Resize the model's memory so it has slots for the new tokens
    model.resize_token_embeddings(len(tokenizer))
    print(f"  [OK] Vocabulary resized to: {len(tokenizer)} tokens.")

    # -------------------------------------------------------------
    # Step 2: Assemble the Packed Sequence (State + 2 Questions)
    # -------------------------------------------------------------
    # Car 0: The Customer State
    state_text = "<|state|> My payouts failed three times. Please fix this."

    # Car 1: Question 1 (Binary Yes/No)
    q1_text = "<|q|> <|noul|> Is this urgent? <|readout|>"

    # Car 2: Question 2 (Multiple Choice)
    q2_text = "<|q|> <|choice|> Which department? <|opt|> Billing <|opt|> Tech <|readout|>"

    print("\n[Step 2] Assembling the text components:")
    print(f"  Car 0 (State): {state_text}")
    print(f"  Car 1 (Q1):    {q1_text}")
    print(f"  Car 2 (Q2):    {q2_text}")

    # Tokenize each part separately so we know their exact lengths
    state_ids = tokenizer.encode(state_text, add_special_tokens=False)
    q1_ids = tokenizer.encode(q1_text, add_special_tokens=False)
    q2_ids = tokenizer.encode(q2_text, add_special_tokens=False)

    # Concatenate all token IDs into one long sequence
    all_token_ids = state_ids + q1_ids + q2_ids
    total_tokens = len(all_token_ids)

    print(f"\n  Length of State: {len(state_ids)} tokens")
    print(f"  Length of Q1:    {len(q1_ids)} tokens")
    print(f"  Length of Q2:    {len(q2_ids)} tokens")
    print(f"  Total packed:    {total_tokens} tokens")

    # -------------------------------------------------------------
    # Step 3: Assign Segment IDs (0 for State, 1 for Q1, 2 for Q2)
    # -------------------------------------------------------------
    # Every token gets a segment tag telling us which car it belongs to
    segments = (
        [0] * len(state_ids) +
        [1] * len(q1_ids) +
        [2] * len(q2_ids)
    )
    segments_tensor = torch.tensor(segments, dtype=torch.long)

    # -------------------------------------------------------------
    # Step 4: Construct the Tree Attention Mask (The Blindfolds)
    # -------------------------------------------------------------
    # We create a 2D grid of size [total_tokens, total_tokens]
    # row i = reader token, col j = target token
    row_idx = torch.arange(total_tokens).unsqueeze(1)  # Column vector [total_tokens, 1]
    col_idx = torch.arange(total_tokens).unsqueeze(0)  # Row vector [1, total_tokens]

    # Rule 1: Causality (cannot look into the future: col <= row)
    causal_mask = col_idx <= row_idx

    # Rule 2: Allowed to look at the State (target segment == 0)
    target_segments = segments_tensor.unsqueeze(0)  # [1, total_tokens]
    reader_segments = segments_tensor.unsqueeze(1)  # [total_tokens, 1]
    can_see_state = target_segments == 0

    # Rule 3: Allowed to look inside the SAME question
    same_branch = reader_segments == target_segments

    # Final Mask: Causal AND (State OR Same Branch)
    tree_mask = causal_mask & (can_see_state | same_branch)

    # -------------------------------------------------------------
    # Step 5: Construct Restarting Position IDs (Order Invariance)
    # -------------------------------------------------------------
    # State tokens get normal positions: 0, 1, 2, ... (S-1)
    S = len(state_ids)
    state_pos = list(range(S))

    # Branch 1 restarts at S: S, S+1, S+2...
    q1_pos = [S + i for i in range(len(q1_ids))]

    # Branch 2 ALSO restarts at S: S, S+1, S+2... (NOT S + len(q1)!)
    q2_pos = [S + i for i in range(len(q2_ids))]

    position_ids = torch.tensor([state_pos + q1_pos + q2_pos], dtype=torch.long)

    print(f"\n[Step 5] Position IDs Preview:")
    print(f"  State positions: {state_pos[:4]} ... {state_pos[-1]}")
    print(f"  Q1 positions:    {q1_pos[0]}, {q1_pos[1]} ... {q1_pos[-1]} (starts at {S})")
    print(f"  Q2 positions:    {q2_pos[0]}, {q2_pos[1]} ... {q2_pos[-1]} (also restarts at {S}!)")

    # -------------------------------------------------------------
    # Step 6: Visualizing the Attention Mask Grid
    # -------------------------------------------------------------
    print("\n[Step 6] Visualizing the 2D Tree Attention Grid:")
    print("('1' = Can see, '.' = BLOCKED blindfold)")
    print("=" * 65)

    tokens_display = [tokenizer.decode([t]).strip() for t in all_token_ids]
    # Print header line
    print("       " + "".join([f"{i % 10}" for i in range(total_tokens)]))
    for i in range(total_tokens):
        row_str = ""
        for j in range(total_tokens):
            row_str += "1" if tree_mask[i, j].item() else "."
        # Label the segments on the left
        seg_label = f"S{segments[i]}"
        print(f"[{seg_label} {i:02d}] {row_str}")

    print("=" * 65)
    print("Notice the big block of dots ('.') where Q2 looks at Q1 -> BLOCKED!")

    # -------------------------------------------------------------
    # Step 7: Run Single Forward Pass through the Model!
    # -------------------------------------------------------------
    print("\n[Step 7] Running forward pass with custom mask and positions...")

    input_ids_tensor = torch.tensor([all_token_ids], dtype=torch.long)

    # Hugging Face transformers expects a 4D attention mask:
    # [batch_size, num_heads, seq_length, seq_length]
    # For additive masking: 0.0 means attend, -inf means blocked
    custom_4d_mask = torch.zeros((1, 1, total_tokens, total_tokens), dtype=torch.float32)
    custom_4d_mask.masked_fill_(~tree_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_tensor,
            attention_mask=custom_4d_mask,
            position_ids=position_ids
        )

    hidden_states = outputs.last_hidden_state
    print(f"  [OK] Forward pass completed! Output shape: {list(hidden_states.shape)}")

    # -------------------------------------------------------------
    # Step 8: Locate the <|readout|> Tokens
    # -------------------------------------------------------------
    readout_id = tokenizer.convert_tokens_to_ids("<|readout|>")
    readout_indices = [idx for idx, t_id in enumerate(all_token_ids) if t_id == readout_id]

    print("\n[Step 8] Locating the <|readout|> decision points:")
    print(f"  Q1 <|readout|> is at Token Index: {readout_indices[0]}")
    print(f"  Q2 <|readout|> is at Token Index: {readout_indices[1]}")

    q1_readout_vector = hidden_states[0, readout_indices[0], :]
    q2_readout_vector = hidden_states[0, readout_indices[1], :]

    print(f"  Q1 Thought Vector sample: {[round(x, 3) for x in q1_readout_vector[:4].tolist()]}")
    print(f"  Q2 Thought Vector sample: {[round(x, 3) for x in q2_readout_vector[:4].tolist()]}")

    print("\n" + "=" * 65)
    print("SUCCESS: Tree Attention & Position IDs fully operational!")
    print("We are 100% ready to plug in Phase 3 Readout Heads!")
    print("=" * 65)


if __name__ == "__main__":
    main()
