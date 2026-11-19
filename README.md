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
├── benchmark_speed_comparison.py  # Wall-clock latency benchmark (Tree vs. Sequential vs. JSON)
├── test_story_comprehension.py    # 12-question reading comprehension demo (Open + Choice + NOUL)
├── run_tree_engine.py             # CLI runner for Track 2 equivalence proofs & 11-scenario suite
├── server.py                      # Track 1 FastAPI server + Two-Stage SHA-256 KV cache
├── studio_ui.html                 # Interactive browser UI for Track 1 server
├── demo_tree_attention.py         # ASCII attention mask visualization & branch isolation
├── demo_readout_heads.py          # Multi-head extraction from packed sequence
├── demo_multi_scenarios.py        # Multi-domain readout evaluation demo
├── demo_kv_cache_serving.py       # KV cache reuse across sequential evaluations
├── train_stage_a.py               # Stage A proper scoring rule (BCE + Brier) warmup script
└── train_colab_1_5b.py            # Full Stage A + Stage B LoRA (q_proj, v_proj) + heads training script
docs/
├── MEDIUM_BLOG_POST.md            # Complete technical walkthrough & article
└── JEV_IMPLEMENTATION_GUIDE.md    # Track 1 architecture & training reference
tests/
├── test_schema.py                 # 12 unit tests for schema, key stripping & synthetic billing
└── test_tree_*.py                 # 21 unit tests for tree builder, masks, cache & modes
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

### 2. Run the Unit Test Suites (33 Tests, ~0.2s)

```bash
python tests/test_schema.py
python -m unittest discover -s tests -p "test_tree_*.py"
```

### 3. Run Track 2: Training-Free Tree Engine Demos, Speed Benchmarks & Suites

* **Wall-Clock Speed & Stage-by-Stage Profiler Benchmark (GPU or CPU):**
  ```bash
  # Auto-detects CUDA GPU (float16) or falls back to CPU (float32):
  python scripts/benchmark_speed_comparison.py --model-name Qwen/Qwen2.5-0.5B

  # Or explicitly target GPU or CPU:
  python scripts/benchmark_speed_comparison.py --model-name Qwen/Qwen2.5-0.5B --device cuda --dtype float16
  python scripts/benchmark_speed_comparison.py --model-name Qwen/Qwen2.5-0.5B --device cpu --dtype float32
  ```
* **12-Question Story Comprehension Walkthrough (Open + Constrained + NOUL in 1 Pass):**
  ```bash
  python scripts/test_story_comprehension.py --model-name Qwen/Qwen2.5-1.5B
  ```
* **11-Scenario Multi-Domain Suite & Mathematical Equivalence Proofs:**
  ```bash
  # Full run including float32 tree-vs-sequential equivalence verification (< 1e-3):
  python scripts/run_tree_engine.py --model-name Qwen/Qwen2.5-0.5B

  # Or run on 1.5B (skipping verification to jump straight to the 11 scenarios):
  python scripts/run_tree_engine.py --model-name Qwen/Qwen2.5-1.5B --skip-verify
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

## 📊 Profiling, Speed Benchmarks & Multi-Domain Results

### 1. Stage-by-Stage GPU Profiler ($Q = 12$: 5 Open + 3 Constrained + 4 NOUL on `NVIDIA Quadro P1000`, `Qwen2.5-0.5B`)
Both Constrained Mode (`score_all_constrained_fused`) and Open Mode (`complete_all_open_fused`) execute directly on top of Pass 1's `tree_kv` (`batch_size = 1`, zero KV cache duplication):

| Pipeline Stage (`TreeEngine.predict()`) | What Happens Inside This Stage | GPU Time (`ms`) | Share of Total |
| :--- | :--- | :--- | :--- |
| **Pass 1 (`pack_guided_tree`)** | Encodes shared story + all 12 question branches in 1 4D Tree Attention pass; extracts single-token logits & `tree_kv` | **589.6 ms** (`49 ms/q`) | **29.4%** |
| **Pass 2a (`score_all_constrained_fused`)** | Scores all candidate tokens + $P(\text{stop})$ across all 7 Constrained & NOUL questions in 1 fused pass on `tree_kv` | **354.3 ms** (`51 ms/q`) | **17.7%** |
| **Pass 2b (`complete_all_open_fused`)** | Completes multi-token BPE subwords (`'Ġc' + 'ello'`) + $P(\text{stop})$ across all 5 Open Mode questions on `tree_kv` | **1,062.3 ms** (`212 ms/q`) | **52.9%** |
| **Full `TreeEngine.predict()` Total** | **All 12 questions (5 Open + 3 Choice + 4 NOUL) + 2D OOD scoring** | **2,006.2 ms** (`167 ms/q`) | **100.0%** |

### 2. End-to-End Speed Comparison: Packed Tree vs. Sequential Calls vs. JSON `.generate()`
* **1a. Tree Prefill (`pack_guided_tree`):** Pass 1 shared-trunk forward pass.
* **1b. Full `TreeEngine`:** Complete end-to-end pipeline (Pass 1 + Fused Constrained + Fused Open Mode + $P(\text{stop})$ + OOD).
* **2. $Q$ Sequential Calls:** Standard approach calling `model(story + question_i)` $Q$ separate times.
* **3. JSON `.generate()`:** Autoregressive generation decoding `6` tokens per question (`{"key": "val"}`).

| Hardware & Model ($Q$ Questions) | 1a. Tree Prefill | 1b. Full `TreeEngine` | 2. $Q$ Sequential Calls | 3. JSON `.generate()` | Full Engine Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPU `Quadro P1000` — `0.5B` ($Q = 4$)** | **414.2 ms** (`104 ms/q`) | **1,730.9 ms** (`433 ms/q`) | 1,860.3 ms (`465 ms/q`) | 4,522.8 ms (`24 tok`) | **1.07× vs. Seq** (`2.61× vs. JSON`) |
| **GPU `Quadro P1000` — `0.5B` ($Q = 8$)** | **481.4 ms** (`60 ms/q`) | **1,842.0 ms** (`230 ms/q`) | 3,526.1 ms (`441 ms/q`) | 8,726.1 ms (`48 tok`) | **1.91× vs. Seq** (`4.74× vs. JSON`) |
| **GPU `Quadro P1000` — `0.5B` ($Q = 12$)** | **581.2 ms** (`48 ms/q`) | **1,892.7 ms** (`158 ms/q`) | 4,726.7 ms (`394 ms/q`) | 12,459.7 ms (`72 tok`) | **2.50× vs. Seq** (**6.58× vs. JSON**) |
| **GPU `Quadro P1000` — `1.5B` ($Q = 12$)** | **4,971.3 ms** (`414 ms/q`) | **5,464.3 ms** (`455 ms/q`) | 28,938.3 ms (`2,412 ms/q`) | 17,012.7 ms (`72 tok`) | **5.30× vs. Seq** (**3.11× vs. JSON**) |
| **CPU `float32` — `0.5B` ($Q = 12$)** | **3,620.4 ms** (`302 ms/q`) | **6,714.2 ms** (`560 ms/q`) | 26,225.8 ms (`2,185 ms/q`) | 14,354.6 ms (`72 tok`) | **3.91× vs. Seq** (**2.14× vs. JSON**) |

### 3. Multi-Domain Evaluation Results (`Qwen2.5-0.5B` vs. `Qwen2.5-1.5B`)
| Suite / Category (`35` Scored + `2` OOD Controls = `37` Total) | `Qwen2.5-0.5B` | `Qwen2.5-1.5B` | Notes |
| :--- | :--- | :--- | :--- |
| **12-Question Story Demo (`test_story_comprehension.py`)** | **50.0%** (`6 / 12`) | **100.0%** (`11 / 12` confident + `1` borderline `52.1%`) | 5 Open + 3 Constrained + 4 NOUL (`3.90s` on GPU) |
| **Secondary Domain Suite (`12` questions)** | **83.3%** (`10 / 12`) | **83.3%** (`10 / 12`) | Hospitality, Port Logistics, University, Agriculture |
| **Primary Dev Suite (`23` scored questions)** | **65.2%** (`15 / 23`) | **69.6%** (`16 / 23`) | Adversarial prefix-collision & multi-token edge cases |
| **Constrained Mode (`24` questions)** | **66.7%** (`16 / 24`) | **75.0%** (`18 / 24`) | 17 of 24 require inferential reasoning beyond surface text |
| **Open Mode (`11` questions)** | **81.8%** (`9 / 11`) | **72.7%** (`8 / 11`) | Zero candidate list provided |
| **Extractive Subset (`13` of 35 questions)** | **84.6%** (`11 / 13`) | **69.2%** (`9 / 13`) | Answer word appears verbatim in the situation text |
| **Inferential Subset (`22` of 35 questions)** | **63.6%** (`14 / 22`) | **77.3%** (`17 / 22`) | Answer requires reasoning (`1.5B` jumps +13.7% here) |
| **OOD Assertions (`3` checks in Scenario 7)** | **2 / 3 Passed** | **3 / 3 Passed** | `1.5B` passes all 3 gross & hard-negative OOD checks |

---

## 📖 Documentation & Technical Write-Up

* **[Medium Blog Post & Full Walkthrough](docs/MEDIUM_BLOG_POST.md):** Beginner-friendly analogies, ML fast-track boxes, BPE tokenization traps (`P(stop)` normalization, bound prefixes), live test results, prior art, and limitations.
* **[Track 1 Implementation Guide](docs/JEV_IMPLEMENTATION_GUIDE.md):** Detailed mathematical formulation and Colab training reference for custom readout heads.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
