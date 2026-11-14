# Pada: Prefill-Only Multi-Branch LLM Decision & Evaluation Engine

**Pada** is an open-source, prefill-only evaluation engine in PyTorch that simultaneously answers multiple structured questions—open-ended single-word extraction, multiple-choice classification, binary Yes/No (`noul`) decisions, and continuous/ordinal scores—over a shared context in a **single Tree Attention forward pass**, eliminating autoregressive token generation.

Instead of running sequential `.generate()` loops to spell out JSON punctuation token-by-token, Pada encodes a shared context trunk once, isolates sibling question branches via a 4D **Tree Attention** mask with **restarted position IDs**, and extracts scores directly at the end of the prefill stage.

---

## ⚡ Two Architectural Tracks Included

### Track 1: Jev-Style Custom Readout Heads (`pada/schema/` + `scripts/server.py`)
Inspired by Archer Hume's 10,000-call empirical probe of TypeSafe AI's Jev model:
* **7 Custom Special Tokens:** `<|state|>`, `<|q|>`, `<|opt|>`, `<|readout|>`, `<|noul|>`, `<|choice|>`, `<|score|>`.
* **Zero Key Leakage (`HostModelAdapter`):** Question keys and option keys are stripped on the host CPU and never enter the tokenizer.
* **Three Specialized Readout Heads:**
  * `NoulHead`: Calibrated binary probability $P(\text{True}) \in (0, 1)$.
  * `ChoiceHead` (Pointer Scorer): Scaled dot-product attention (`(k @ q) / sqrt(d)`) comparing `<|readout|>` against any number of `<|opt|>` markers.
  * `UniversalScoreHead`: Continuous `[0, 1]` rating slider rescaled dynamically to custom ranges (`1 to 5`, `1 to 10`) and binomial ordinal distributions.
* **Two-Stage SHA-256 Prefix KV Cache:** Encodes and caches the shared `state` once; evaluates incoming question branches in a single pass.

### Track 2: Training-Free, Few-Shot Anchored `lm_head` Tree Engine (`pada/tree_engine/`)
Built to overcome the closed-world and domain-specific training requirements of custom readout heads:
* **Direct `lm_head` Probing (Zero Fine-Tuning):** Uses the base model's pretrained `lm_head.weight` (`[151936, d]` in Qwen2.5, where `tie_word_embeddings=True` on `0.5B` and `1.5B`) anchored by a domain-disjoint few-shot prefix (`GUIDED_ANCHOR_PROMPT`).
* **Unified Mixed-Mode Execution:** Evaluates **Open Mode** (no candidates), **Constrained Mode** (multiple choice), and **NOUL Mode** (`candidates=["yes", "no"]`) together in one packed Tree Attention pass.
* **Open-Vocabulary Content-Word Mask (`content_word_mask`):** Filters Qwen2.5's `151,936` vocabulary tokens down to `50,780` space-prefixed English content words, with a `bound_prefix_mask` (`538` stems, `forbid_bound_stop=True`) that forces truncated BPE stems (`'Ġloc'`, `'Ġemerg'`, `'Ġc'`) to attach their subword continuation (`'locust'`, `'emergency'`, `'cello'`).
* **Termination-Boundary Scoring ($P(\text{stop})$) & 2D OOD Detection:** Multiplies candidate joint probabilities by $P(\text{stop} \mid \text{candidate})$ to resolve BPE prefix collisions (`'cardiology'` vs. `'cardiac'`) and tracks raw probability mass + premise log-likelihood to flag out-of-distribution (`OOD`) questions.

---

## 📁 Repository Structure

```text
pada/
├── schema/                        # Track 1: Formal request/response contract & host adapter
│   ├── request.py                 # JevRequest (state + choice/noul/score questions)
│   ├── response.py                # JevResponse (probabilities, expected score, usage)
│   ├── internal.py                # HostModelAdapter (zero key leakage)
│   ├── billing.py                 # Synthetic token billing formula
│   └── limits.py                  # Contract constraints
├── tree_engine/                   # Track 2: Training-free lm_head Tree Attention engine
│   ├── engine.py                  # TreeEngine pipeline (Open, Constrained, NOUL)
│   ├── tree_builder.py            # 4D Tree Attention mask & restarted position_ids
│   ├── lexicon.py                 # Vocabulary masks, bound-prefix guard, Snowball stemmer
│   ├── constrained.py             # Fused teacher-forced candidate & P(stop) scoring
│   ├── open_mode.py               # Branched subword continuation & active-row pruning
│   ├── calibration.py             # 2D zero-pass OOD threshold calibration
│   ├── cache_utils.py             # Zero-deepcopy DynamicCache slicing & cloning
│   ├── suites.py                  # 11-scenario multi-domain evaluation suites
│   └── verification.py            # Mathematical tree-vs-sequential equivalence proofs
scripts/
├── test_story_comprehension.py    # 12-question reading comprehension demo (Open + Choice + NOUL)
├── run_tree_engine.py             # CLI runner for Track 2 equivalence proofs & 11-scenario suite
├── server.py                      # Track 1 FastAPI server + Two-Stage SHA-256 KV cache
├── studio_ui.html                 # Interactive browser UI for Track 1 server
├── demo_tree_attention.py         # ASCII attention mask visualization & branch isolation
├── demo_readout_heads.py          # Multi-head extraction from packed sequence
├── demo_multi_scenarios.py        # Multi-domain readout evaluation demo
├── demo_kv_cache_serving.py       # KV cache reuse across sequential evaluations
└── train_stage_a.py               # Proper scoring rule (BCE + Brier) training script
docs/
├── MEDIUM_BLOG_POST.md            # Complete technical walkthrough & article
└── JEV_IMPLEMENTATION_GUIDE.md    # Track 1 architecture & training reference
tests/
├── test_schema.py                 # 12 unit tests for schema, key stripping & synthetic billing
└── test_tree_*.py                 # 18 unit tests for tree builder, masks, cache & modes
```

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and set up a Python 3.11+ virtual environment:

```bash
git clone https://github.com/Akintunde102/pada.git
cd pada
python -m venv .venv

# Activate on Linux / macOS:
source .venv/bin/activate
export PYTHONPATH="."

# Or activate on Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
# $env:PYTHONPATH = "."

pip install torch transformers accelerate peft fastapi uvicorn nltk
```

### 2. Run the Unit Test Suites (30 Tests, ~0.1s)

```bash
python tests/test_schema.py
python -m unittest discover -s tests -p "test_tree_*.py"
```

### 3. Run Track 2: Training-Free Tree Engine Demos & Suites

* **12-Question Story Comprehension Walkthrough (Open + Constrained + NOUL in 1 Pass):**
  ```bash
  python scripts/test_story_comprehension.py --model-name Qwen/Qwen2.5-1.5B
  ```
* **11-Scenario Multi-Domain Suite & Mathematical Equivalence Proofs:**
  ```bash
  # Full run including float32 tree-vs-sequential equivalence verification (< 1e-3):
  python scripts/run_tree_engine.py --model-name Qwen/Qwen2.5-0.5B

  # Or skip verification to run the 11 scenarios directly:
  python scripts/run_tree_engine.py --model-name Qwen/Qwen2.5-0.5B --skip-verify
  ```

### 4. Run Track 1: Jev-Style FastAPI Server & Architectural Demos

* **Launch the FastAPI Server & Web Studio (`http://127.0.0.1:8000`):**
  ```bash
  python scripts/server.py
  ```
* **Inspect Tree Attention Masking & KV Cache Reuse:**
  ```bash
  python scripts/demo_tree_attention.py
  python scripts/demo_kv_cache_serving.py
  ```

---

## 📖 Documentation & Technical Write-Up

* **[Medium Blog Post & Full Walkthrough](docs/MEDIUM_BLOG_POST.md):** Beginner-friendly analogies, ML fast-track boxes, BPE tokenization traps (`P(stop)` normalization, bound prefixes), live test results, prior art, and limitations.
* **[Track 1 Implementation Guide](docs/JEV_IMPLEMENTATION_GUIDE.md):** Detailed mathematical formulation and Colab training reference for custom readout heads.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
