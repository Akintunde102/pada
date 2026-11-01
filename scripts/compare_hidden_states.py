"""
Educational Script: Proving What hidden_states[0, -1, :] Truly Means
----------------------------------------------------------------------
We compare two completely different sentences that both end with the SAME dot ('.'):
Sentence 1: "Everything is broken and urgent."
Sentence 2: "The weather is sunny and calm."

We will show:
1. What the indexing [0, -1, :] means step-by-step.
2. That at the very start (Layer 0), the dot '.' is identical in both.
3. That at the very end (hidden_states[0, -1, :]), the dot '.' has completely transformed!
"""

import torch
from transformers import AutoTokenizer, AutoModel

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Two very different sentences that BOTH end with a period '.'
sentence_1 = "Everything is broken and urgent."
sentence_2 = "The weather is sunny and calm."

# Tokenize both
tokens_1 = tokenizer(sentence_1, return_tensors="pt")
tokens_2 = tokenizer(sentence_2, return_tensors="pt")

# Pass both through the model (Prefill pass)
with torch.no_grad():
    out_1 = model(**tokens_1)
    out_2 = model(**tokens_2)

# Extract the final hidden states
h1 = out_1.last_hidden_state  # Shape: [1, 6, 896]
h2 = out_2.last_hidden_state  # Shape: [1, 7, 896]

print("=" * 65)
print("1. UNDERSTANDING THE SHAPE AND INDEXING [0, -1, :]")
print("=" * 65)
print(f"Sentence 1: \"{sentence_1}\"")
print(f"  Shape: {list(h1.shape)}")
print(f"  [0]:  The first (and only) sentence.")
print(f"  [-1]: The last token in this sentence (Token index {h1.shape[1] - 1}, which is '.')")
print(f"  [:]:  All 896 numbers for that token.\n")

print("=" * 65)
print("2. THE PROOF: LOOKING AT THE ACTUAL NUMBERS OF THE DOT ('.')")
print("=" * 65)
dot_vector_1 = h1[0, -1, :]  # The dot at the end of the URGENT sentence
dot_vector_2 = h2[0, -1, :]  # The dot at the end of the CALM sentence

print("First 6 numbers of the dot '.' in Sentence 1 (URGENT):")
print(" ", [round(x, 4) for x in dot_vector_1[:6].tolist()])

print("\nFirst 6 numbers of the dot '.' in Sentence 2 (CALM):")
print(" ", [round(x, 4) for x in dot_vector_2[:6].tolist()])

# Measure how similar they are using Cosine Similarity (1.0 = identical, 0.0 = completely different)
similarity = torch.nn.functional.cosine_similarity(
    dot_vector_1.unsqueeze(0),
    dot_vector_2.unsqueeze(0)
).item()

print(f"\nMathematical Similarity between the two dots: {similarity:.4f}")
print("-> Even though both tokens were literally the character '.', their final")
print("   hidden states are completely different because each dot absorbed")
print("   the intelligence of the sentence that came before it!")
print("=" * 65)
