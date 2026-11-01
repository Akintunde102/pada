# Pada: High-Performance Prefill-Only Multi-Branch LLM Evaluation Engine

**Pada** is an open-source, prefill-only evaluation architecture that simultaneously answers multiple structured questions, classification tasks, and ordinal scores over a shared context in a **single forward pass**—completely eliminating autoregressive token generation.

Traditional LLM workflows query models sequentially or run token-by-token generation for every question, incurring massive latency and cost. Pada solves this by processing a single shared context trunk and branching out queries via **Tree Attention**, extracting calibrated probabilities directly from specialized readout heads at the end of the prefill stage.

---

## ⚡ Key Architectural Innovations

### 1. Prefill-Without-Decode (Zero Generation Latency)
Standard chat APIs generate tokens one at a time via autoregressive decoding loops. Pada operates strictly in the **prefill stage**:
* The shared state/context is ingested once.
* All queries and candidate options are evaluated simultaneously in parallel.
* Hidden state vectors at key boundary positions are mapped directly into calibrated probability distributions.

### 2. Tree Attention Masking (Prefix Sharing with Branch Isolation)
In multi-question evaluation, sibling questions must never influence or contaminate each other's representations:
* **Shared Trunk**: Every question attends to the shared state context ($j \le i$ and $\text{segment} = 0$).
* **Isolated Branches**: Tokens within Question $A$ cannot attend to Question $B$ and vice versa ($\text{segment}_i == \text{segment}_j$).
* Evaluates $N$ questions concurrently in a single batch forward pass without $N \times$ duplicate state computation.

### 3. Restarted Position IDs (Order Invariance)
To guarantee that the order in which questions appear has zero effect on their computed probabilities:
* The shared context occupies positions $0 \dots S-1$.
* Each question branch independently restarts its position counter at $S, S+1, \dots$.
* Evaluation is mathematically invariant to question ordering.

### 4. Specialized Readout Heads
Rather than parsing free-form generated text:
* **`ChoiceHead` (Pointer Scorer)**: Projects question readout vectors and option representations into a shared metric space using scaled dot-product attention (`(k @ q) / sqrt(d)`). Dynamically handles any arbitrary number of choices.
* **`NoulHead` (Binary Calibrated Confidence)**: Computes a scalar probability $p \in [0, 1]$ via calibrated linear projection and sigmoid.
* **`ScoreHead` (Ordinal Scoring)**: Evaluates $K$-point Likert scales, calculating exact probability distributions and expected scores $\mathbb{E}[S] = \sum_{k=1}^K k \cdot P(k)$.

### 5. Zero Key Leakage
The API contract cleanly decouples external dictionary keys from internal model tensors. Question and option keys are stripped prior to model serialization and remapped on the host, preventing key names from leaking into or biasing model representations.

---

## 📁 Repository Structure

```
pada/
├── schema/                 # Formal request/response contract & serialization
│   ├── request.py          # PadaRequest (state + choice/noul/score questions)
│   ├── response.py         # PadaResponse (calibrated probabilities & usage)
│   ├── internal.py         # Key stripping and host-model adaptation
│   ├── billing.py          # Synthetic token accounting
│   └── limits.py           # Contract constraints (state chars, max questions)
scripts/
├── inspect_backbone.py     # Inspect prefill hidden states of the base model
├── compare_hidden_states.py# Demonstrates causal context accumulation in vectors
├── demo_tree_attention.py  # ASCII attention mask visualization & branch isolation
├── demo_readout_heads.py   # Multi-head extraction from packed sequence
├── demo_multi_scenarios.py # Multi-domain evaluation (FinTech, Triage, Fraud)
├── demo_kv_cache_serving.py# KV cache reuse across sequential evaluations
├── train_stage_a.py        # Proper scoring rule training for readout heads
└── server.py               # FastAPI inference server
tests/
└── test_schema.py          # Verification test suite for schema contracts
```

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and set up a Python 3.11+ environment:

```bash
git clone https://github.com/<your-username>/pada.git
cd pada

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch transformers accelerate fastapi uvicorn
```

### 2. Run Tests

Verify schema contracts and key stripping:

```bash
python tests/test_schema.py
```

### 3. Run Architectural Demonstrations

* **Inspect Tree Attention Masking & Readout:**
  ```bash
  python scripts/demo_tree_attention.py
  ```

* **Evaluate Multi-Domain Scenarios:**
  ```bash
  python scripts/demo_multi_scenarios.py
  ```

* **Inspect Shared KV Cache Serving:**
  ```bash
  python scripts/demo_kv_cache_serving.py
  ```

---

## 📊 Verification & Training

* **Stage A (Readout Heads on Frozen Backbone):**
  Trained on diverse multi-domain datasets with soft targets and option-order shuffling.
  * Penalty loss dropped from **8.101 → 0.984**.
  * Demonstrates instant pointer routing for support categorization, fraud detection, and triage with zero decoding overhead.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
