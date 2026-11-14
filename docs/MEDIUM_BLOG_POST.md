# Stop Paying LLMs to Spell JSON: Building a Jev-Style Prefill Decision Engine—And Pivoting to a Training-Free 1-Pass Tree Engine

*A practical guide to building a prefill-only multi-branch LLM decision engine in PyTorch—written for both curious newcomers and ML systems engineers.*

---

> **🗺️ How to Read This Post**
> * **New to Machine Learning?** Read straight through! Every concept is explained first in plain English using everyday analogies before showing a single line of code.
> * **Already know Transformers & PyTorch?** Look for the **⚡ ML Fast-Track** boxes at the top of each section—they give you the exact tensor shapes, equations, and architectural mechanics so you can skip the beginner analogies whenever you want.

---

## 1. What Triggered This? (And Why We Built It)

On September 15, 2026, **TypeSafe AI** (founded by RLHF co-creator Diogo Almeida) launched **Jev** in early access—describing it as the first public "System One" model built for fast, structured decisions rather than chat, trained with **RLCD (Reinforcement Learning for Calibrated Decisions)**.

Shortly after launch, an independent researcher (**archerhume**) ran a **10,000-call black-box probe** over two weeks against the TypeSafe API—measuring upstream latencies, token-billing formulas, question-order permutations, and 200-option boundary behaviors ([summary coverage](https://www.besthub.dev/articles/reverse-engineering-jev-10k-api-calls-expose-closed-source-model-architecture-27085f944485)).

Together, TypeSafe's design and Hume's probe highlighted two major flaws in how production software typically uses Large Language Models (LLMs) for operational decisions like fraud screening, ticket routing, medical triage, and content moderation:

### Flaw #1: Paying for Spelling, Not Thinking
Standard LLMs work **autoregressively**—they generate text one token (word piece) at a time in a sequential loop.
If your backend asks an LLM to route a customer support ticket and return `{"department": "payments", "urgent": true}`, the GPU has to execute a separate forward pass for `{`, `"`, `department`, `"`, `:`, `"`, `payments`, `"`, `,`...

Even with a KV cache, every single generated character requires streaming billions of model weights from GPU memory into the compute cores. You are paying for memory-bandwidth bottlenecks just so a neural network can spell punctuation marks.

### Flaw #2: Verbalized Confidence Is Fake Probability
If you ask a standard chatbot *"How confident are you?"* and it replies `"90% confident"`, that is **not** a calibrated 90% mathematical probability. It just means the model assigned high probability to generating the string `"90%"`. Standard post-training (RLHF) rewards models for sounding decisive, which routinely causes probability distributions to collapse toward overconfident extremes.

### What Is Hume's vs. What Is Ours
Because Jev is closed-weight and proprietary, we cannot claim to replicate TypeSafe's exact internal weights. Instead, it helps to separate what Hume observed from what we designed:
* **Hume's Probe Findings:** Jev's latency is invariant from 2 to 200 options (a 200-option query billed at 1,911 `output_tokens` runs in the same time as a 2-option query, proving `output_tokens` is a synthetic billing formula rather than a decode count); state prefixes are shared while sibling questions are isolated with zero cross-talk; options interact in-context (listwise scoring); and outputs are calibrated probabilities across three primitives: `choice`, `score`, and `noul` (short for a *Bernoulli* yes/no trial). Hume also *speculated* that Jev uses a Sparse Mixture-of-Experts (MoE) backbone.
* **Our Engineering (Two Open-Source Tracks on `Qwen2.5-0.5B` & `1.5B`):**
  1. **Track 1 (`Pada` — A Jev-Style Rebuild):** Our own implementation of a prefill-only engine using 7 custom special tokens (`<|state|>`, `<|q|>`, `<|opt|>`, `<|readout|>`, `<|noul|>`, `<|choice|>`, `<|score|>`), a host-side key-stripping adapter (`HostModelAdapter`), and three custom readout heads (`NoulHead`, a scaled dot-product `ChoiceHead` pointer scorer, and a continuous `UniversalScoreHead`).
  2. **Track 2 (`Nameless` — Training-Free, Few-Shot Anchored `lm_head` Tree Engine):** When our custom heads hit a generalizability wall on unseen domains, we built a second engine that bypasses custom heads and probes the model's native **`lm_head`** directly—evaluating multiple open-vocabulary, multiple-choice, and boolean questions in a single tree-attention pass without any fine-tuning.

---

## 2. Concept 1: Prefill Without Decode (Stopping Before the Loop)

> **⚡ ML Fast-Track:** Causal transformers process all prompt tokens simultaneously during the compute-bound **prefill** phase via lower-triangular attention masking, whereas autoregressive **decoding** is memory-bandwidth bound ($O(N)$ sequential forward passes streaming full weight matrices per token). We drop the autoregressive decode loop completely, halt immediately after prefill, and extract hidden representations $h_{\text{last}} \in \mathbb{R}^d$ (or slice logits at key boundary indices) in a single pass.

### 🌱 The Beginner Picture: Reading Once vs. Hauling the Printing Press for Every Letter
Imagine an expert inspector reading a one-page incident report with three questions at the bottom:
* **Reading the page (Prefill):** Their eyes take in the whole page and the questions in one sweep. By the time they reach the question mark at the end of each question, their brain has already formed the answer.
* **Typing out JSON (Decode):** In a standard LLM, even though the model saves its notes from reading the page (the **KV cache**), generating each character of the reply (`{`, `"`, `a`, `n`, `s`...) requires hauling the model's entire multi-gigabyte brain out of storage into the processor **one character at a time**.

**Both of our engines stop right after the "Reading" step.** At the final token of each question, the model produces a list of numbers (a **hidden state vector**) that already encodes its decision. We read that vector directly and format the JSON on the CPU.

```
Standard LLM (.generate):
[Prefill Prompt] ──> [Load Weights for "{"] ──> [Load Weights for '"'] ──> ... (20–100 sequential passes)

Our Prefill-Only Engine:
[Prefill Context + All Questions in Parallel] ──> [Extract Scores Directly] (1 pass, 0 words generated)
```

---

## 3. Concept 2: Tree Attention & Restarted Position IDs

> **⚡ ML Fast-Track:** Given a shared state prefix of length $S$ (segment $0$) and $M$ question branches (segments $1 \dots M$), we pack all tokens into a single sequence of length $N = S + \sum_{k=1}^M Q_k$ and construct a 4D additive attention mask $A \in \{0, -\infty\}^{1 \times 1 \times N \times N}$ where token $i$ attends to token $j$ iff:
> $$j \le i \quad \land \quad (\text{seg}_j = 0 \lor \text{seg}_i = \text{seg}_j)$$
> To prevent RoPE (Rotary Position Embedding) drift and ensure question-order invariance, every branch $k > 0$ resets its position ID counter to start at $S$: $\text{pos}_i = S + (i - \text{start}_k)$. In float32, packed tree passes are numerically equivalent to separate sequential forward passes (`max |tree_logits - seq_logits| < 1e-3`, differing only by floating-point reduction order in attention kernels).

### 🌱 The Beginner Picture: One Textbook, Isolated Exam Cubicles
Suppose you have **1 story** and **10 different questions** to ask about it:
* If you send 10 separate requests, the model has to read the same story 10 times.
* If you paste all 10 questions into one normal prompt, **Question 2 can see Question 1**. Asking *"Is this customer furious?"* in Question 1 can bias the model when it answers *"Which department should handle this?"* in Question 2—and swapping the order of questions changes the answers.

To read the story **once**, evaluate all 10 questions **in one pass**, and keep them **completely blind to each other**, we combine two techniques:
1. **Tree Attention (Privacy Dividers):** We place the shared story in Segment `0`, Question 1 in Segment `1`, Question 2 in Segment `2`, and so on. Our attention mask lets every question look back at Segment `0` (the story) and at itself, **but blocks every question from seeing any sibling question**.
2. **Restarted Position Rulers (`position_ids`):** Transformers stamp a position number (`0, 1, 2...`) on every token. If the story uses positions `0` through `49`, we number Question 1 starting at `50, 51, 52...` **and we also number Question 2 starting at `50, 51, 52...`!** Every question sits at the exact same distance from the story, making the outputs numerically invariant to question ordering.

```
Position IDs:   [0, 1, ..., S-1]   [S, S+1, S+2]   [S, S+1, S+2]   [S, S+1, S+2]
Segments:       [  Segment 0   ]   [ Segment 1 ]   [ Segment 2 ]   [ Segment 3 ]
                ┌──────────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
Packed Input:   │ Shared Story │◄──┤Question 1 │   │Question 2 │   │Question 3 │
                └──────────────┘   └───────────┘   └─────┬─────┘   └─────┬─────┘
                       ▲                                 │               │
                       └─────────────────────────────────┴───────────────┘
                        (All branches see Segment 0; branches NEVER see each other)
```

### 💻 Code Example: Building the Tree Attention Mask & Position IDs

```python
import torch

def build_tree_inputs(state_ids: list[int], branch_id_lists: list[list[int]]):
    S = len(state_ids)
    all_ids = list(state_ids)
    segments = [0] * S
    pos_ids = list(range(S))  # State uses positions 0 .. S-1

    for branch_idx, b_ids in enumerate(branch_id_lists, start=1):
        all_ids.extend(b_ids)
        segments.extend([branch_idx] * len(b_ids))
        # Every branch restarts its position ruler at S!
        pos_ids.extend(range(S, S + len(b_ids)))

    N = len(all_ids)
    seg_t = torch.tensor(segments)
    rows = torch.arange(N).unsqueeze(1)
    cols = torch.arange(N).unsqueeze(0)

    # Rule: causal (cols <= rows) AND (looking at state OR looking at same branch)
    allowed = (cols <= rows) & ((seg_t.unsqueeze(0) == 0) | (seg_t.unsqueeze(1) == seg_t.unsqueeze(0)))

    # Convert boolean table to additive attention mask: 0.0 for allowed, -inf for blocked
    mask_4d = torch.zeros((1, 1, N, N)).masked_fill_(~allowed, float("-inf"))
    return torch.tensor([all_ids]), mask_4d, torch.tensor([pos_ids])
```

---

## 4. Track 1: Our Jev-Style Rebuild ("Pada")—Custom Readout Heads & Two-Stage KV Caching

> **⚡ ML Fast-Track:** In `scripts/server.py` and `pada/schema/`, we designed a Jev-inspired architecture with 7 custom special tokens (`<|state|>`, `<|q|>`, `<|opt|>`, `<|readout|>`, `<|noul|>`, `<|choice|>`, `<|score|>`), a host-side key-stripping layer (`HostModelAdapter` so dictionary keys never enter the tokenizer), and three lightweight readout heads over final-layer hidden states $h \in \mathbb{R}^d$:
> * **`NoulHead`:** $P(\text{True}) = \sigma(W_{\text{noul}}\,\text{LN}(h_{\text{readout}}) / T_{\text{noul}})$
> * **`ChoiceHead` (Pointer Scorer):** Scaled dot-product attention between the question readout vector and post-option `<|opt|>` vectors: $\text{logit}_i = \frac{(W_k\,\text{LN}(h_{\text{opt}_i})) \cdot (W_q\,\text{LN}(h_{\text{readout}}))}{\sqrt{d}}$, supporting arbitrary option counts.
> * **`UniversalScoreHead`:** Continuous sigmoid projection rescaled dynamically to $[V_{\min}, V_{\max}]$ and mapped to $K$-level ordinal distributions via a binomial expansion.

### 🌱 The Beginner Picture: Three Custom-Trained Judges
In Track 1 (`scripts/server.py`), we added special marker tags after each option (`<|opt|>`) and at the end of each question (`<|readout|>`), then trained three small neural "judges" (Readout Heads) on top of `Qwen2.5-1.5B` (with a rank-8 LoRA adapter on `q_proj` and `v_proj`):
1. **`NoulHead` (The Yes/No Judge):** Projects the `<|readout|>` vector into a single probability between `0.0` and `1.0`.
2. **`ChoiceHead` (The Pointer Judge):** Compares the question's `<|readout|>` vector against each option's `<|opt|>` vector using a scaled dot-product. Because it points to option markers rather than fixed output slots, it can score 2 options or 200 options dynamically.
3. **`UniversalScoreHead` (The Rating Slider):** Outputs a smooth continuous rating in `(0.0, 1.0)` that rescales to any requested range (`1 to 5`, `1 to 10`).

### 💻 Code Example: The Three Readout Heads (`scripts/server.py`)

```python
import math
import torch
import torch.nn as nn

class NoulHead(nn.Module):
    """Binary Yes/No probability judge."""
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h)).squeeze(-1)


class ChoiceHead(nn.Module):
    """Pointer Scorer: compares question readout against any number of <|opt|> vectors."""
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.scale = math.sqrt(d)

    def forward(self, h_readout: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        q = self.W_q(self.norm(h_readout))
        k = self.W_k(self.norm(h_opts))
        return (k @ q) / self.scale


class UniversalScoreHead(nn.Module):
    """Continuous [0, 1] rating slider."""
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h)).squeeze(-1)
```

We trained these heads using **Cross-Entropy + Brier Loss** $(\hat{p} - y)^2$ (a strictly proper scoring rule) and fitted scalar temperatures ($T_{\text{noul}}=1.50, T_{\text{choice}}=2.00, T_{\text{score}}=1.50$). We also built a **Two-Stage SHA-256 Prefix KV Cache** in `scripts/server.py`: Stage 1 encodes and caches the `state` once, and Stage 2 evaluates incoming question branches against the cached state KV tensors.

---

## 5. The Turning Point: What Happened When We Tested Track 1 Live

We launched our trained `Qwen2.5-1.5B` server (`python scripts/server.py`), whose readout heads and LoRA adapter had been trained in Colab on **customer product reviews**.

First, we tested it on an in-domain product review:
> **State:** *"This blender broke on the second day! Plastic gears stripped completely and customer service ignored my emails. Total waste of money."*

It performed cleanly across all three heads in a single pass:
* **`sentiment` (`ChoiceHead`):** `neg`: **87.2%**, `neu`: **12.8%**, `pos`: **0.04%**
* **`positive` (`NoulHead`):** `p_true`: **12.0%** (88% confident it is *not* positive)
* **`rating` (`UniversalScoreHead`):** `0.58 / 4.0` (skewed to the lowest tier: `[53.7%, 36.1%, 9.1%, 1.0%, 0.04%]`)

**Then we sent it an out-of-domain query—customer support ticket routing:**
> **State:** *"My payouts have failed three times. The bank says everything is fine. Can someone fix this?"*
> * **Options:** `payments` (*"Payout failures and payment processing"*), `account` (*"Login and account access"*), `other` (*"Something else"*)

And the model returned:
* `account`: **88.99%** ❌
* `payments`: **5.75%**
* `other`: **5.25%**

### Our Hypothesis for Why Track 1 Failed Out-of-Domain
Why did a 1.5B model confuse payout failures with login access?
1. **Domain-Specific Readout Weights:** Unlike the pretrained language model head, `ChoiceHead` and `NoulHead` start as newly initialized projection matrices (`W_q`, `W_k`, `proj`). Because our Colab training set only covered product review sentiment, we hypothesize the readout projections overfit to sentiment features rather than general instruction-following (though prompt serialization differences and base-model priors may also contribute).
2. **Closed-World Constraint:** Even with massive multi-domain training data, a pointer head over `<|opt|>` markers can **only** choose among options explicitly written into the prompt. It can never answer an open-ended factual question (like *"Which university did Amara graduate from?"*) without listing candidates in advance.

That led us to ask: **Can we get single-pass tree evaluation across both constrained choices AND open-ended questions without training any custom heads at all?**

---

## 6. Track 2: The "Nameless" Pivot (Training-Free, Few-Shot Anchored `lm_head` Tree Engine)

> **⚡ ML Fast-Track:** Every causal LLM already has a pretrained readout matrix: `lm_head.weight` $\in \mathbb{R}^{V \times d}$ ($V = 151,936$ in Qwen2.5). Notably, both `Qwen2.5-0.5B` and `Qwen2.5-1.5B` set `tie_word_embeddings = True`, meaning `lm_head.weight` is tied directly to the input token embedding matrix (`embed_tokens.weight`).
> * **Constrained & NOUL Mode:** We prepend a static, domain-disjoint few-shot anchor (`GUIDED_ANCHOR_PROMPT`), pack all question branches into a 4D tree mask, and score candidate words (including binary `["yes", "no"]`) by evaluating their token sequences and multiplying by termination probability $P(\text{stop} \mid \text{candidate})$.
> * **Open Mode:** We mask `lm_head` logits with a precomputed `content_word_mask` ($50,780$ space-prefixed word tokens across lowercase, Title-case, and uppercase), enforce a `bound_prefix_mask` continuation guard ($538$ bound BPE prefixes with `forbid_bound_stop=True`) so truncated stems (`'Ġemerg'`, `'Ġloc'`, `'Ġc'`) must attach their subword suffix, and group inflected forms via Snowball stemming.

### Prior Art vs. What Is New Here
Scoring candidate label words via language model logits is a well-established paradigm:
* **Verbalizers & Pattern-Exploiting Training (Schick & Schütze, 2021):** Mapped classification labels to single vocabulary words in masked/causal LMs.
* **Log-Likelihood Evaluation Harnesses (EleutherAI `lm-evaluation-harness`):** Score multiple-choice candidates by teacher-forcing continuation tokens and comparing log-probabilities.
* **Constrained Decoding Libraries (`Outlines`, `Guidance`, `SGLang`):** Enforce regex/CFG schemas via token masks during generation.

**What our Track 2 (`pada/tree_engine/`) combines and contributes:**
1. **Unified Mixed-Mode Tree Pass:** Packing open-vocabulary extraction, constrained multiple-choice, and boolean `yes`/`no` questions into a **single shared-trunk 4D Tree Attention pass** with restarted position IDs.
2. **Open-Vocabulary Content-Word Masking (`content_word_mask`):** Filtering Qwen2.5's `151,936` vocabulary tokens down to `50,780` clean space-prefixed English content words so open-ended questions return a standalone word without a decode loop.
3. **Bound-Prefix Continuation Guard (`bound_prefix_mask`):** Preventing BPE subword stems (`'Ġloc'`, `'Ġindemn'`, `'Ġc'`) from prematurely terminating in Open Mode.
4. **Termination-Boundary Normalization ($P(\text{stop})$):** Multiplying candidate joint probabilities by the probability of emitting a stop boundary token right after the word to fix BPE prefix collisions.

### 🌱 The Beginner Picture: Using the Dictionary the Model Already Knows
At the top of `Qwen2.5` sits a built-in dictionary table called the **Language Model Head (`lm_head`)** with **151,936 rows**. Most common English words already have their own dedicated row in this table:
* Row `33531` is `' billing'`
* Row `33671` is `' urgent'`
* Row `20965` is `' refund'`
* Row `9834` is `' yes'` / Row `902` is `' no'`

In fact, on `Qwen2.5-0.5B` and `1.5B`, this output dictionary is **tied (`tie_word_embeddings=True`)** to the model's input reading dictionary—meaning the vector the model uses to *output* a word is the exact same vector it uses to *understand* that word.

---

### Solving 4 Real BPE Tokenization Traps Along the Way

When you extract single words from `lm_head` without a multi-sentence decode loop, four tokenization traps appear immediately:

#### Trap 1: The "Sentence Starter" Trap (Why Few-Shot Anchoring Is Required)
If you feed a raw base model `"Situation: ... Question: What is the issue? Answer:"` truly zero-shot, the #1 token in `lm_head` is `' that'` (**62.5%**) or `' the'`, because the base model wants to start a full clause (*"The issue is that..."*).
* **The Fix (Few-Shot Anchoring):** Our engine is **training-free, but few-shot anchored**. We prepend `GUIDED_ANCHOR_PROMPT` (`pada/tree_engine/constants.py`), which contains two short, domain-disjoint examples (Astronomy and Pastry Baking: `telescope`, `star`, `gamma`, `no`, `tart`, `butter`, `chef`, `yes`). By the time the model reaches `One-word answer:`, its hidden state is locked into emitting a single standalone word.

#### Trap 2: BPE Prefix Collisions (`"cardiac"` vs. `"cardiology"`)
In Qwen's tokenizer, `' cardiac'` is 1 token, while `' cardiology'` splits into `' cardi' + 'ology'`. Comparing raw token probabilities biases the model toward shorter prefixes or incomplete words.
* **The Fix (Termination-Boundary Scoring via $P(\text{stop})$):** A candidate word is only complete if the model wants to **stop** after it (by emitting `.`, `\n`, or `<|im_end|>`). We multiply each candidate's token probability by $P(\text{stop} \mid \text{candidate})$:
  $$\text{Score}(\text{word}) = P(\text{tokens}) \times P(\text{stop boundary} \mid \text{tokens})$$
  *(Note: This fixes tokenization length and prefix bias so candidates compete fairly; however, unlike Track 1, we have not yet fitted post-hoc temperature scaling or measured Expected Calibration Error [ECE] on Track 2, so these normalized shares should be interpreted as relative model scores rather than verified calibrated probabilities).*

#### Trap 3: Bound BPE Stems (`"locust"` = `['Ġloc', 'ust']`) & The `'cello'` Bug We Caught!
What if the model wants to output a multi-token word in Open Mode, like `"locust"` (`['Ġloc', 'ust']`)?
We precompute a `bound_prefix_mask` (`538` stems in Qwen2.5): if a root token is a truncated BPE stem like `'Ġloc'`, `forbid_bound_stop=True` prevents it from stopping at step 0 and runs a 1-token KV-cached continuation (~12× faster than Pass 1) to attach `'ust'` $\rightarrow$ **`"locust"`**.

**How We Caught the `'cello'` Bug During Testing:** When we first ran our reading-comprehension story demo asking *"Which musical instrument does Daniel enjoy playing?"*, `Qwen2.5-1.5B` output **`"cell"`** instead of **`"cello"`**!
Inspecting Qwen's tokenizer revealed why:
```python
>>> tokenizer.convert_ids_to_tokens(tokenizer.encode(" cello"))
['Ġc', 'ello']
```
Qwen splits `' cello'` into a 1-letter prefix **`'Ġc'`** (`272`) + **`'ello'`** (`4791`). Our vocabulary filter in `lexicon.py` originally had `if len(clean) >= 2:`, which masked out `'Ġc'`! Forced to pick a $\ge 2$-letter token, the model fell back to `'Ġcell'`.
Updating `lexicon.py` to allow `len(clean) >= 1` (excluding `"a"` and `"i"`) in `content_word_mask` while marking all 1-letter prefixes as `bound_prefix_mask = True` (so `'Ġc'` can never stop as `"c"` and *must* attach a continuation) immediately resolved `'Ġc' + 'ello'` into **`"cello"` (94.7% score)**.

#### Trap 4: Out-of-Distribution (OOD) Nonsense Questions
If a softmax is forced to pick between `["mortgage", "parliament", "spreadsheet"]` on a medical emergency, relative shares still sum to 100% (`mortgage: 97.7%`).
* **The Fix (2D Zero-Pass OOD Detection):** We track **Raw Probability Mass** ($\sum P(\text{candidates})$ before normalizing) and **Question Premise Log-Likelihood**. On the `"mortgage"` test, raw mass drops to **`7.38e-08`**, flagging `is_ood = True`.

---

## 7. Walkthrough Demo & Development Suite Results

### Demo 1: Literature Reading Comprehension Walkthrough (`scripts/test_story_comprehension.py`)
*(Methodology note: This 12-question story is an **illustrative walkthrough demo**, not a held-out benchmark—in fact, we used this exact story to diagnose the `'Ġc' + 'ello'` prefix bug and to add the two boolean `yes`/`no` examples to `GUIDED_ANCHOR_PROMPT`).*

> **Story:** *"Amara Okafor graduated from Harvard with a degree in architecture before marrying Daniel Mensah in Lagos. Last autumn, the couple moved to Toronto so Amara could lead a museum renovation project. Even though Daniel enjoys playing the cello in his free time, he has been furious all week because the airline lost his luggage during the flight. Meanwhile, Amara's younger sister Chidinma still attends Oxford, where she studies robotics and volunteers at a local hospital on weekends."*

All **12 questions** run together in a single packed Tree Attention pass on `Qwen/Qwen2.5-1.5B` (comparing against `0.5B`):

| Question Key | Question Asked | Mode | Expected | `0.5B` Output (Score) | `1.5B` Output (Score) | `1.5B` Eval |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `amara_school_open` | Which university did Amara graduate from? | **OPEN** | `Harvard` | `harvard` (91.8%) | `harvard` (**95.7%**) | ✅ **PASS** |
| `spouse_first_name_open` | First name of the man Amara married? | **OPEN** | `Daniel` | `daniel` (94.5%) | `daniel` (**97.9%**) | ✅ **PASS** |
| `current_city_open` | Which city did the couple move to? | **OPEN** | `Toronto` | `toronto` (99.1%) | `toronto` (**99.3%**) | ✅ **PASS** |
| `sister_school_open` | Which university does her sister attend? | **OPEN** | `Oxford` | `oxford` (86.1%) | `oxford` (**98.0%**) | ✅ **PASS** |
| `instrument_open` | Which musical instrument does Daniel play? | **OPEN** | `cello` | `piano` (32.2%) | `cello` (**94.7%**) | ✅ **PASS** |
| `amara_profession_choice` | What field did Amara earn her degree in? | **CONSTRAINED** | `architecture` | `architecture` (99.7%) | `architecture` (**100.0%**) | ✅ **PASS** |
| `wedding_city_choice` | In which city did they get married? | **CONSTRAINED** | `Lagos` | `Toronto` (98.5%) | `Lagos` (**71.2%**) | ✅ **PASS** |
| `sister_major_choice` | What subject does Chidinma study? | **CONSTRAINED** | `robotics` | `robotics` (97.8%) | `robotics` (**99.9%**) | ✅ **PASS** |
| `noul_daniel_angry` | Is Daniel angry right now? | **NOUL (Y/N)** | `yes` | `no` (59.2%) | `yes` (**81.2%**) | ✅ **PASS** |
| `noul_amara_doctor` | Is Amara a medical doctor? | **NOUL (Y/N)** | `no` | `yes` (72.8%) | `no` (**97.4%**) | ✅ **PASS** |
| `noul_luggage_lost` | Did the airline lose Daniel's luggage? | **NOUL (Y/N)** | `yes` | `no` (88.5%) | `yes` (**73.4%**) | ✅ **PASS** |
| `noul_sister_in_toronto` | Does Chidinma attend school in Toronto? | **NOUL (Y/N)** | `no` | `yes` (71.7%) | `no` (**52.1%**) | ⚠️ **Borderline Pass** *(near 50/50 coin flip)* |
| **Summary Accuracy** | | | | **6 / 12 (50.0%)** | **12 / 12 (100.0%)** | *(11 high-conf + 1 borderline)* |

---

### Evaluation 2: The 11-Scenario Multi-Domain Suite (`Qwen2.5-0.5B` on CPU)
In `pada/tree_engine/suites.py`, we also evaluated the compact **`Qwen2.5-0.5B`** model on CPU across 11 scenarios (7 Dev scenarios + 4 Held-Out scenarios across Clinical Triage, Financial Fraud, E-Commerce, Cybersecurity, Legal Contracts, Terse Complaints, OOD Controls, Hospitality, Port Logistics, University, and Agriculture). *(Note: While the anchor prompt was kept strictly domain-disjoint, these suites were used during development iteration).*

| Suite Metric (`Qwen2.5-0.5B` on CPU) | Accuracy | Notes |
| :--- | :--- | :--- |
| **Held-Out Suite (4 Scenarios)** | **83.3%** (`10 / 12`) | Hospitality, Port Logistics, University, Agriculture |
| **Extractive Questions** | **84.6%** (`11 / 13`) | Pulling factual entities directly from context |
| **Open Mode Questions** | **81.8%** (`9 / 11`) | The 2 misses were valid synonyms (`urgent` vs `critical`, `broken` vs `damaged`) |
| **Constrained Mode Questions** | **66.7%** (`16 / 24`) | Selecting from provided candidate lists |
| **Dev Suite (7 Adversarial Scenarios)** | **65.2%** (`15 / 23`) | Hard prefix-collision and multi-token edge cases |
| **Average CPU Latency** | **444.9 ms / question** | 37 questions evaluated in `16.46s` on laptop CPU |

---

## 8. How to Run and Test Everything Yourself

The full codebase is open-source under the **MIT License**:

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

### Way 1: Run the 12-Question Story Comprehension Demo
```bash
python scripts/test_story_comprehension.py --model-name Qwen/Qwen2.5-1.5B
```

### Way 2: Run the 11-Scenario Multi-Domain Suite & Equivalence Proofs
```bash
# Run with mathematical equivalence verification (checks max |tree_logits - seq_logits| < 1e-3):
python scripts/run_tree_engine.py --model-name Qwen/Qwen2.5-0.5B

# Or skip verification to jump straight to the 11 scenarios:
python scripts/run_tree_engine.py --model-name Qwen/Qwen2.5-0.5B --skip-verify
```

### Way 3: Launch the Track 1 FastAPI Server & Interactive Web Studio
```bash
python scripts/server.py
```
Open `http://127.0.0.1:8000` in your browser or POST to `http://127.0.0.1:8000/v1/evaluate`.

### Way 4: Run the Unit Test Suites (30 Tests)
```bash
python tests/test_schema.py
python -m unittest discover -s tests -p "test_tree_*.py"
```

---

## 9. Honest Limitations & What's Left to Be Done

1. **No ECE / Reliability Diagram Calibration Yet on Track 2 (`Nameless`):**
   * Track 1 (`Pada`) trains readout heads with Brier + Cross-Entropy loss and fits temperature scalars ($T$). Track 2 (`Nameless`) uses raw `lm_head` logits normalized by $P(\text{stop})$. While $P(\text{stop})$ removes tokenization length bias, we have **not** yet measured Expected Calibration Error (ECE) or fitted temperature scaling on a large labeled validation set for Track 2.
2. **Single-Word Output Boundary:**
   * Track 2's Open Mode is designed for **single-word** answers (`"Harvard"`, `"cardiology"`, `"locust"`). Multi-word entities (*"New York City"*) stop at the first complete word in Open Mode.
3. **Token Casing Sensitivity in Constrained Candidates:**
   * In Scenario 2 (SIM-swapping fraud), our candidate list specified lowercase `"sim"`, while the model placed 32.1% of its unconstrained probability mass on uppercase `"SIM"`. Marginalizing over lowercase, Title-case, and UPPERCASE BPE surface forms for each candidate is an easy win left to implement.
4. **Small Dense Models (`0.5B` / `1.5B`) vs. Larger / MoE Models:**
   * On indirect multi-hop negation (*"Chidinma attends Oxford... Does Chidinma attend school in Toronto?"*), `0.5B` failed (`71.7% yes`) and `1.5B` barely passed (`52.1% no`). We hypothesize that scaling to `Qwen2.5-7B` or an open Sparse MoE like `Qwen3-30B-A3B` or `DeepSeek-V3` will improve multi-hop boolean reasoning, though we have only benchmarked `0.5B` and `1.5B` so far.
5. **Custom FlashInfer / FlexAttention Kernels:**
   * We currently materialize dense 4D masks (`[1, 1, N, N]`) for PyTorch SDPA. Replacing this with **PyTorch FlexAttention `BlockMask`** or **FlashInfer Cascade Attention** will skip masked branch blocks at the CUDA kernel level.

---

## 10. References & Further Reading

* **Jev & Archer Hume's Probe:**
  * [TypeSafe AI Launch Coverage (Let's Data Science)](https://letsdatascience.com/news/typesafe-ai-launches-jev-decision-model-889a38c0)
  * [Reverse-Engineering Jev: 10k API Calls Expose Architecture (BestHub Summary of Hume's Probe)](https://www.besthub.dev/articles/reverse-engineering-jev-10k-api-calls-expose-closed-source-model-architecture-27085f944485)
  * [Jev: A New Kind of AI Model Built for Decisions (Eden AI)](https://www.edenai.co/post/jev-a-new-kind-of-ai-model-built-for-decisions-not-conversation)
* **Prior Art on Verbalizers & Log-Likelihood Scoring:**
  * Schick & Schütze (2021), *"Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference"* (EACL 2021).
  * Gao et al., *"A Framework for Few-Shot Language Model Evaluation"* (`lm-evaluation-harness`).
* **Calibration & Latent Knowledge in LLMs:**
  * Kadavath et al. (Anthropic), *"Language Models (Mostly) Know What They Know"* (arXiv:2207.05221).
  * Burns et al., *"Discovering Latent Knowledge in Language Models Without Supervision"* (ICLR 2023).
  * Glenn W. Brier (1950), *"Verification of forecasts expressed in terms of probability"*.
* **Tree Attention & Prefix Sharing Systems:**
  * Zheng et al., *"SGLang: Efficient Execution of Structured Language Model Programs"* (RadixAttention).
  * Cai et al., *"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"*.
  * PyTorch Blog, *"FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention"*.
