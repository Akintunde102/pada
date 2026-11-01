"""
Educational Script: Inspecting the Jev Backbone Model and Hidden States
------------------------------------------------------------------------
This script demonstrates how Phase 1 works in practice:
1. Load a modern causal language model (Qwen 2.5) without the text-generation head.
2. Tokenize a short sentence into numeric IDs.
3. Perform a single 'Prefill' forward pass.
4. Inspect the resulting 'Hidden States' (the raw thought vectors).
"""

import sys
# PyTorch is the underlying math library that does matrix operations on tensors
import torch

# From the transformers library, we import:
# - AutoTokenizer: automatically picks the right text-to-number translator for our model.
# - AutoModel: loads the base neural network (the 'brain') WITHOUT the text-speaking head.
from transformers import AutoTokenizer, AutoModel


def main():
    print("=" * 60)
    print("Phase 1: Loading the Backbone and Inspecting Hidden States")
    print("=" * 60)

    # 1. Choose which model to inspect.
    # We use Qwen2.5-0.5B for our quick initial test (downloads in seconds, ~900 MB).
    # You can change this to "Qwen/Qwen2.5-1.5B" anytime; the code works identically!
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\n[Step 1] Loading tokenizer and model: {model_name}...")

    # AutoTokenizer.from_pretrained downloads the vocabulary and grammar rules.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # AutoModel.from_pretrained loads the transformer layers.
    # Notice: We use AutoModel, NOT AutoModelForCausalLM.
    # This ensures we only load the brain, completely omitting the vocabulary generator (LM head).
    model = AutoModel.from_pretrained(model_name)

    # Set the model to evaluation mode ('eval').
    # This turns off training features like dropout so the model behaves deterministically.
    model.eval()
    print("[OK] Model and tokenizer successfully loaded into memory!")

    # 2. Prepare a sample input sentence.
    # In Jev, this is like the customer support ticket state.
    text_input = "My payouts have failed three times. Please fix this."
    print(f"\n[Step 2] Translating English sentence:")
    print(f"  Input text: \"{text_input}\"")

    # The tokenizer takes our string and converts each word/subword into a number.
    # 'return_tensors="pt"' tells it to return PyTorch tensors (multi-dimensional arrays).
    encoded = tokenizer(text_input, return_tensors="pt")

    # input_ids contains the list of numeric IDs representing each token.
    token_ids = encoded["input_ids"]
    print(f"  Token IDs: {token_ids.tolist()}")
    print(f"  Total tokens created: {token_ids.shape[1]}")

    # Let's inspect what each number actually corresponds to in plain English:
    tokens_as_words = [tokenizer.decode([single_id]) for single_id in token_ids[0]]
    print(f"  Tokens mapped back to words: {tokens_as_words}")

    # 3. Perform the single 'Prefill' pass through the neural network.
    print(f"\n[Step 3] Running single forward pass through the transformer...")

    # 'torch.no_grad()' tells PyTorch: 'We are just reading, not training.'
    # This saves computer memory and makes the computation much faster.
    with torch.no_grad():
        # We pass the tokenized dictionary (**encoded) into the model.
        outputs = model(**encoded)

    # 4. Extract the 'last_hidden_state'.
    # This is the goldmine in Jev: the raw mathematical representations computed by the final layer.
    hidden_states = outputs.last_hidden_state

    # The shape tells us the dimensions of the data:
    # [batch_size, sequence_length, hidden_dimension]
    batch_size, sequence_length, hidden_dimension = hidden_states.shape

    print(f"\n[Step 4] Hidden State Analysis:")
    print(f"  Batch size:        {batch_size} (We sent 1 sentence)")
    print(f"  Sequence length:   {sequence_length} tokens")
    print(f"  Hidden dimension:  {hidden_dimension} numbers per token")
    print(f"  Full tensor shape: {list(hidden_states.shape)}")

    # Let's inspect the very last token's representation.
    # In Jev, this last position is where our readout head will plug in!
    last_token_vector = hidden_states[0, -1, :]
    print(f"\n[Step 5] The 'Thought Vector' for the final token ({tokens_as_words[-1]!r}):")
    print(f"  First 5 numbers of this 896-dimension vector: {last_token_vector[:5].tolist()}")
    print("  (This dense vector summarizes the context of the entire message!)")

    print("\n" + "=" * 60)
    print("Phase 1 Verification Complete: Backbone operates in prefill-only mode!")
    print("=" * 60)


if __name__ == "__main__":
    main()
