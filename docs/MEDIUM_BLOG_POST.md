# Answering Multiple Questions in One LLM Pass: How I Built an Open-Source Jev Engine in PyTorch

*Using Tree Attention and the model's built-in vocabulary table (`lm_head`) to get multiple-choice, yes/no, and single-word answers without generating text token-by-token—written for both curious newcomers and ML systems engineers.*

> **📦 Full Open-Source Repository (MIT License):** [**github.com/Akintunde102/pada**](https://github.com/Akintunde102/pada)  
> Every file, class, function, and benchmark referenced in this article links directly to the exact source file and line numbers in the repository above so you can inspect or run the code as you read.

---

> **🗺️ How to Read This Post**
> * **New to Machine Learning?** Read straight through! Every concept is explained first in plain English using everyday analogies before showing a single line of code.
> * **Already know Transformers & PyTorch?** Look for the **⚡ ML Fast-Track** boxes at the top of each section—they give you the exact tensor shapes, equations, and architectural mechanics so you can skip the beginner analogies whenever you want.

---

## 1. What Triggered This? (And Why I Built It)

On September 15, 2026, **TypeSafe AI** (founded by CEO and RLHF co-creator Diogo Almeida alongside co-founders Erik Gafni and Sasha Sheng) launched **Jev** in early access—describing it as the first public "System One" model built for fast, structured decisions rather than chat, trained with **RLCD (Reinforcement Learning for Calibrated Decisions)**.

Shortly after launch, an independent researcher under the handle **`archerhume`** ran a **10,000-call black-box probe** over two weeks against the TypeSafe API—measuring upstream latencies, reported token usage fields, option-order and position variations, and 2-option vs. 200-option behaviors ([BestHub republished summary of the probe](https://www.besthub.dev/articles/reverse-engineering-jev-10k-api-calls-expose-closed-source-model-architecture-27085f944485)).

Together, TypeSafe's design and `archerhume`'s probe highlighted two common pain points in how production software uses Large Language Models (LLMs) for operational decisions like fraud screening, ticket routing, medical triage, and content moderation:

### Pain Point #1: Paying for Spelling Instead of Shared-Trunk Branching
Standard LLM workflows often ask a model to generate structured JSON (`{"department": "payments", "urgent": true}`) one token (word piece) at a time in a sequential loop. Even with a KV cache, every generated token requires streaming billions of model weights from GPU memory into the compute cores.

* **What about `max_tokens=1` + `logprobs` + Prefix Caching?** Experienced ML engineers know you don't *have* to generate full JSON to get label probabilities—you can set `max_tokens=1` with `logprobs=True` and rely on prefix caching (via vLLM or SGLang) to share the document state across requests.
* **Where single-token probing still falls short:** First, evaluating $Q$ questions still requires dispatching $Q$ separate requests or sequences rather than packing isolated branches into one shared-trunk forward pass. Second, when candidate words split into multiple BPE tokens (like `' cardiology'` $\rightarrow$ `['Ġcard', 'iology']`), `max_tokens=1` cannot score the full word or verify that the word actually terminates ($P(\text{stop} \mid \text{candidate})$). Third, for open-ended questions without a fixed candidate list, `max_tokens=1` gets trapped by sentence starters (`' that'`, `' the'`) or truncated BPE stems (`'Ġloc'`, `'Ġc'`).

### Pain Point #2: Verbalized Confidence Is Not Guaranteed to Be Calibrated
When applications ask a chat model to output a text string like `"90% confident"`, that string is generated token-by-token—it is not automatically guaranteed to reflect a calibrated 90% empirical accuracy. While some research finds that verbalized confidence can be reasonably calibrated on certain RLHF models, TypeSafe introduced **RLCD** as a training method to directly optimize output probability distributions rather than human chat preferences, exposing those probabilities across three output primitives: `choice` (categorical options), `score` (ordered rubrics), and `noul` (binary yes/no decisions). *(Independent evaluations of Jev's calibration vary by domain: `archerhume` reported an Expected Calibration Error [ECE] of `0.0313` on MMLU, while an [independent synthetic support-ticket evaluation by Rajesh Beri cited by Sébastien Dubois](https://www.dsebastien.net/reinforcement-learning-for-calibrated-decisions-rlcd/) reported an ECE of `0.107`).*

### Prior Art, `archerhume`'s Probe, and My Two Tracks in `Pada`
Because Jev is closed-weight and proprietary, I am not claiming to replicate TypeSafe's internal weights. Furthermore, several open-source projects have already explored Jev-style reconstructions:
* **TypeSafe's Docs & `archerhume`'s Probe:** TypeSafe's API documentation states that questions are evaluated independently and that decisions are billed on input tokens only. In `archerhume`'s 10,000-call probe, a 200-option query reported `output_tokens=1911` in the usage object at the same latency as a 2-option query (showing `output_tokens` is a synthetic reported usage field rather than an autoregressive decode count), state prefixes were shared across questions, options interacted in-context (listwise scoring), and `archerhume` hypothesized a Sparse Mixture-of-Experts (MoE) backbone.
* **Existing Open-Source Jev Clones:** Community projects such as **[Kev](https://github.com/jaredpalmer/kev)** (`jaredpalmer/kev` by Jared Palmer, which pairs a Qwen base model + LoRA adapter + pointer head + shared-state attention masking; see also [The Menon Lab analysis](https://themenonlab.blog/blog/kev-open-source-decision-model)), **[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev)** (`TheoLeeCJ/SemIf-OpenJev`, formerly `openjev`), **[sarvam-jev](https://github.com/SAGAR-TAMANG/sarvam-jev/wiki)** (`SAGAR-TAMANG/sarvam-jev`), **[`kyegomez/open-jev`](https://github.com/kyegomez/open-jev)**, **Laya**, **Nimble** (which reads candidate-token logits), and **jevos** have explored shared-state prefill and readout architectures (see the [Lilting architecture comparison](https://lilting.ch/en/articles/jev-clones-architecture-comparison)).
* **What I Built in My Repository ([`Akintunde102/pada`](https://github.com/Akintunde102/pada) on `Qwen2.5-0.5B` & `1.5B`):**
  1. **Track 1 (Custom Readout Heads — [`pada/schema/`](https://github.com/Akintunde102/pada/tree/main/pada/schema) + [`scripts/server.py`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py)):** Similar in spirit to `Kev`, I built a prefill-only engine with 7 custom special tokens (`<|state|>`, `<|q|>`, `<|opt|>`, `<|readout|>`, `<|noul|>`, `<|choice|>`, `<|score|>`), a host-side key-stripping adapter ([`HostModelAdapter` in `pada/schema/internal.py#L73`](https://github.com/Akintunde102/pada/blob/main/pada/schema/internal.py#L73)), and three custom readout heads ([`NoulHead`, `ChoiceHead`, and `UniversalScoreHead` in `scripts/server.py#L59-L97`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L59-L97)), trained via [`scripts/train_colab_1_5b.py`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py).
  2. **Track 2 (Training-Free `lm_head` Tree Engine & `Nameless` Open Mode — [`pada/tree_engine/`](https://github.com/Akintunde102/pada/tree/main/pada/tree_engine)):** When my custom heads hit an out-of-domain wall, I built a second engine inside [`pada/tree_engine/engine.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/engine.py#L37) that bypasses custom heads and probes the model's native **`lm_head`** directly—combining a shared-trunk 4D Tree Attention pass ([`pack_guided_tree` in `pada/tree_engine/tree_builder.py#L11`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11)) with a fused KV-cached continuation pass to evaluate closed-world multiple-choice and boolean questions, plus an extended **Open Mode (`Nameless` in [`pada/tree_engine/open_mode.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py#L193))** for open-vocabulary single-word extraction without any fine-tuning.

---

## 2. Concept 1: Prefill Without Decode (Stopping Before the Loop)

> **⚡ ML Fast-Track:** Causal transformers process all prompt tokens simultaneously during the compute-bound **prefill** phase via lower-triangular attention masking, whereas autoregressive **decoding** is memory-bandwidth bound ($O(N)$ sequential forward passes streaming full weight matrices per generated token). I drop the autoregressive JSON decode loop, extract hidden representations $h_{\text{last}} \in \mathbb{R}^d$ (or read logits at branch boundaries) at the end of prefill, and run at most a short, batched KV-cached continuation for multi-token candidates or stems.

### 🌱 The Beginner Picture: Reading Once vs. Hauling the Printing Press for Every Token
Imagine an inspector reading a one-page incident report with three questions at the bottom:
* **Reading the page (Prefill):** Their eyes take in the whole page and the questions in one sweep. By the time they reach the end of each question, their brain has already formed the answer.
* **Typing out JSON (Decode):** In a standard LLM, even though the model saves its notes from reading the page (the **KV cache**), generating each token (word piece) of a JSON reply (`{`, `"`, `department`, `"`...) requires hauling the model's entire multi-gigabyte weight matrix out of memory into the processor **one token at a time**.

**My engines stop right after the "Reading" step.** At the final token of each question, the model produces a list of numbers (a **hidden state vector**) that already encodes its decision. I read that vector directly and format the JSON on the CPU.

```
Standard LLM (.generate):
[Prefill Prompt] ──> [Load Weights for Token 1] ──> [Load Weights for Token 2] ──> ... (Sequential Passes)

My Tree Prefill Engine:
[Prefill Context + All Questions in Parallel] ──> [Extract Scores Directly] (0 JSON Tokens Decoded)
```

---

## 3. Concept 2: Tree Attention & Restarted Position IDs ([`pada/tree_engine/tree_builder.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11-L64))

> **⚡ ML Fast-Track:** Given a shared state prefix of length $S$ (segment $0$) and $M$ question branches (segments $1 \dots M$), [`pack_guided_tree()` in `pada/tree_engine/tree_builder.py#L11-L64`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11-L64) packs all tokens into a single sequence of length $N = S + \sum_{k=1}^M Q_k$ and constructs a 4D additive attention mask $A \in \{0, -\infty\}^{1 \times 1 \times N \times N}$ where token $i$ attends to token $j$ iff:
> $$j \le i \quad \land \quad (\text{seg}_j = 0 \lor \text{seg}_i = \text{seg}_j)$$
> Because RoPE (Rotary Position Embeddings) encodes relative token distance, packing branches sequentially would place later branches at a larger relative distance from the state. To make every branch's relative distance to the state identical and guarantee question-order invariance, every branch $k > 0$ resets its position ID counter to start at $S$: $\text{pos}_i = S + (i - \text{start}_k)$. In float32, packed tree passes are numerically equivalent to separate sequential forward passes (`max |tree_logits - seq_logits| < 1e-3`, verified in [`pada/tree_engine/verification.py#L33-L93`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/verification.py#L33-L93)).
> *(Implementation note: In [`pada/tree_engine/tree_builder.py#L59`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L59), masked positions are filled with `torch.finfo(dtype).min` to avoid `NaN`s across `float32`, `float16`, and `bfloat16`).*

### 🌱 The Beginner Picture: One Textbook, Isolated Exam Cubicles
Suppose you have **1 story** and **12 different questions** to ask about it:
* If you send 12 separate requests, the model re-encodes the shared prefix 12 times.
* If you paste all 12 questions into one normal prompt, **Question 2 can see Question 1**. Asking *"Is this customer furious?"* in Question 1 can bias the model when it answers *"Which department should handle this?"* in Question 2—and swapping the order of questions changes their distance from the story.

To read the story **once**, evaluate all 12 questions **in one pass**, and keep them **completely blind to each other**, I combine two techniques in [`pada/tree_engine/tree_builder.py#L11-L64`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11-L64):
1. **Tree Attention (Privacy Dividers):** I place the shared story in Segment `0`, Question 1 in Segment `1`, Question 2 in Segment `2`, and so on. My attention mask lets every question look back at Segment `0` (the story) and at itself, **but blocks every question from seeing any sibling question**.
2. **Restarted Position Rulers (`position_ids`):** Transformers stamp a position number (`0, 1, 2...`) on every token. If the story uses positions `0` through `49`, I number Question 1 starting at `50, 51, 52...` **and I also number Question 2 starting at `50, 51, 52...`!** Every question sits at the exact same relative distance from the story, regardless of branch order.

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

### 💻 Code Example: Building the Tree Attention Mask & Position IDs ([`pada/tree_engine/tree_builder.py#L11-L64`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11-L64))

```python
import torch

def build_tree_inputs(state_ids: list[int], branch_id_lists: list[list[int]], dtype=torch.float32):
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

    # Convert boolean table to 4D additive attention mask: 0.0 for allowed, finfo.min for blocked
    mask_4d = torch.zeros((1, 1, N, N), dtype=dtype).masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)
    return torch.tensor([all_ids]), mask_4d, torch.tensor([pos_ids])
```

---

## 4. Track 1: Custom Readout Heads & Two-Stage KV Caching ([`scripts/server.py`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py))

> **⚡ ML Fast-Track:** In [`scripts/server.py`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py), [`scripts/train_colab_1_5b.py`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py), and [`pada/schema/`](https://github.com/Akintunde102/pada/tree/main/pada/schema), Track 1 uses 7 custom special tokens (`<|state|>`, `<|q|>`, `<|opt|>`, `<|readout|>`, `<|noul|>`, `<|choice|>`, `<|score|>`), a host-side key-stripping layer ([`HostModelAdapter` in `pada/schema/internal.py#L73`](https://github.com/Akintunde102/pada/blob/main/pada/schema/internal.py#L73) so dictionary keys never enter the tokenizer), and three lightweight readout heads ([`scripts/server.py#L59-L97`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L59-L97)) over final-layer hidden states $h \in \mathbb{R}^d$. Each head `forward()` outputs **raw logits**, to which [`evaluate_jev_request()` in `scripts/server.py#L216-L315`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L216-L315) applies temperature scaling ($1 / T$) and `sigmoid` / `softmax`:
> * **`NoulHead` ([`scripts/server.py#L59-L66`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L59-L66)):** $\text{logit} = W_{\text{noul}}\,\text{LN}(h_{\text{readout}}) \implies P(\text{True}) = \sigma(\text{logit} / T_{\text{noul}})$
> * **`ChoiceHead` ([`scripts/server.py#L69-L80`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L69-L80)):** Given $h_{\text{readout}} \in \mathbb{R}^d$ and option matrix $H_{\text{opts}} \in \mathbb{R}^{M \times d}$, $\text{logits} = \frac{(W_k\,\text{LN}(H_{\text{opts}})) \cdot (W_q\,\text{LN}(h_{\text{readout}}))}{\sqrt{d}} \implies P = \text{softmax}(\text{logits} / T_{\text{choice}})$.
> * **`UniversalScoreHead` ([`scripts/server.py#L83-L95`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L83-L95)):** $\text{logit} = W_{\text{score}}\,\text{LN}(h_{\text{readout}}) \implies s_{\text{norm}} = \sigma(\text{logit} / T_{\text{score}}) \in (0, 1)$, rescaled to $[V_{\min}, V_{\max}]$ and mapped to $K$-level ordinal distributions via [`continuous_to_ordinal_distribution()` in `scripts/server.py#L200-L213`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L200-L213).

### 🌱 The Beginner Picture: Three Custom-Trained Judges
In Track 1 ([`scripts/server.py`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py) and [`scripts/train_colab_1_5b.py`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py)), I added special marker tags after each option (`<|opt|>`) and at the end of each question (`<|readout|>`), then trained three small neural "judges" (Readout Heads) on top of `Qwen2.5-1.5B` (with a rank-8 LoRA adapter on `q_proj` and `v_proj`):
1. **`NoulHead` (The Yes/No Judge):** Projects the `<|readout|>` vector into a raw score (logit), which a sigmoid function converts into a probability between `0.0` and `1.0`.
2. **`ChoiceHead` (The Pointer Judge):** Compares a single question's `<|readout|>` vector (`[d]`) against that question's `<|opt|>` vectors (`[M, d]`) using a scaled dot-product. Because it points to option markers rather than fixed output slots, it can score 2 options or 200 options dynamically.
3. **`UniversalScoreHead` (The Rating Slider):** Outputs a raw score that a sigmoid converts into a smooth rating in `(0.0, 1.0)` and rescales to any requested range (`1 to 5`, `1 to 10`).

### 💻 Code Example: The Three Readout Heads ([`scripts/server.py#L59-L97`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L59-L97))

```python
import math
import torch
import torch.nn as nn

class NoulHead(nn.Module):
    """Binary Yes/No judge (returns raw logit; caller applies sigmoid(logit / T_noul))."""
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h)).squeeze(-1)


class ChoiceHead(nn.Module):
    """Unbatched Pointer Scorer: compares 1 question vector [d] against M option vectors [M, d].
    Caller applies softmax(logits / T_choice).
    """
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
    """Continuous rating judge (returns raw logit; caller applies sigmoid(logit / T_score))."""
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h)).squeeze(-1)
```

In [`scripts/train_colab_1_5b.py#L171-L260`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py#L171-L260), I trained these heads using **Cross-Entropy + Brier Loss** $(\hat{p} - y)^2$ (a strictly proper scoring rule) on 1,000 Amazon product reviews (Stage A on frozen representations, followed by Stage B joint LoRA + head tuning) and applied scalar temperatures ($T_{\text{noul}}=1.50, T_{\text{choice}}=2.00, T_{\text{score}}=1.50$). I also built a **Two-Stage SHA-256 Prefix KV Cache** in [`get_state_memory()` (`scripts/server.py#L172-L180`)](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L172-L180): Stage 1 encodes and caches the `state` once, and Stage 2 evaluates incoming question branches against the cached state KV tensors.

---

## 5. The Turning Point: What Happened When I Tested Track 1 Live

I launched my trained `Qwen2.5-1.5B` server (`python scripts/server.py`), whose readout heads and LoRA adapter had been trained on **customer product reviews**.

First, I tested it on an in-domain product review:
> **State:** *"This blender broke on the second day! Plastic gears stripped completely and customer service ignored my emails. Total waste of money."*

It performed cleanly across all three heads in a single pass:
* **`sentiment` (`ChoiceHead`):** `neg`: **87.2%**, `neu`: **12.8%**, `pos`: **0.04%**
* **`positive` (`NoulHead`):** `p_true`: **12.0%** (88% confident it is *not* positive)
* **`rating` (`UniversalScoreHead`):** `0.58 / 4.0` (skewed to the lowest tier: `[53.7%, 36.1%, 9.1%, 1.0%, 0.04%]`)

**Then I sent it an out-of-domain query—customer support ticket routing:**
> **State:** *"My payouts have failed three times. The bank says everything is fine. Can someone fix this?"*
> * **Options:** `payments` (*"Payout failures and payment processing"*), `account` (*"Login and account access"*), `other` (*"Something else"*)

And the model returned:
* `account`: **88.99%** ❌
* `payments`: **5.75%**
* `other`: **5.25%**

### My Hypothesis for Why Track 1 Failed Out-of-Domain
Why did a 1.5B model confuse payout failures with login access?
1. **Narrowly Trained Readout Weights:** Unlike the pretrained language model head, `ChoiceHead` and `NoulHead` are newly trained projection matrices (`W_q`, `W_k`, `proj`). Because my training script ([`scripts/train_colab_1_5b.py`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py)) only trained on product review sentiment, I hypothesize that the readout projections overfit to sentiment features rather than general instruction-following.
2. **Closed-World Constraint:** Even with massive multi-domain training data, a pointer head over `<|opt|>` markers can **only** choose among options explicitly written into the prompt. It can never answer an open-ended factual question (like *"Which university did Amara graduate from?"*) without listing candidates in advance.

That led me to ask: **Can I evaluate both constrained choices AND open-ended questions over a shared tree trunk without training any custom heads at all?**

---

## 6. Completing the Closed-World Jev System Without Custom Heads (`Constrained` & `NOUL` on `lm_head`)

> **⚡ ML Fast-Track:** Every causal LLM already has a pretrained readout matrix: `lm_head.weight` $\in \mathbb{R}^{V \times d}$ ($V = 151,936$ in Qwen2.5). Both `Qwen2.5-0.5B` and `Qwen2.5-1.5B` set `tie_word_embeddings = True`, meaning `lm_head.weight` is tied directly to `embed_tokens.weight`.
> To run closed-world Jev decisions (`choice` and binary `noul` `yes`/`no`) without training custom heads, [`TreeEngine.predict()` in `pada/tree_engine/engine.py#L112-L295`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/engine.py#L112-L295) executes:
> * **Pass 1 (Shared-Trunk Tree Pass — [`pack_guided_tree` in `pada/tree_engine/tree_builder.py#L11-L64`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11-L64)):** Packs [`GUIDED_ANCHOR_PROMPT` (`pada/tree_engine/constants.py#L51-L63`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L51-L63) + `Situation: <state>` (Segment $0$, length $S$) and all $M$ isolated question branches (`Question: <instr> One-word answer:`, Segments $1 \dots M$, positions restarted at $S$) into one forward pass with `use_cache=True`. For single-token binary `noul` (`' yes'` = `9834`, `' no'` = `902`) or single-token choices, `outputs.logits[0, readout_idx, :]` immediately yields $\log P(t_1 \mid \text{state}, q_k)$.
> * **Pass 2 (Fused Stage-2 Candidate & Boundary Pass — [`score_all_constrained_fused` in `pada/tree_engine/constrained.py#L128-L223`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L128-L223)):** To score multi-token candidates (`' cardiology'` $\rightarrow$ `['Ġcard', 'iology']`) and read $P(\text{stop} \mid \text{candidate})$ after each candidate's final token, I pack all candidate token sequences across all constrained questions directly on top of Pass 1's `tree_kv` (`batch_size = 1`) using a 4D continuation mask (`[1, 1, T, P + T]`), run **one fused teacher-forced pass**, and crop `tree_kv` back to $P$ via [`crop_cache()` in `pada/tree_engine/cache_utils.py#L47-L51`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/cache_utils.py#L47-L51).

### 🌱 The Beginner Picture: Using the Dictionary the Model Already Knows
Before going beyond Jev, I first needed to solve Jev's closed-world tasks (`choice` and `noul`) in a way that works across **any** domain without training custom heads.

At the top of `Qwen2.5` sits a built-in dictionary table called the **Language Model Head (`lm_head`)** with **151,936 rows**. Thousands of English words already have their own dedicated single-token row in this table:
* Row `33531` is `' billing'`
* Row `33671` is `' urgent'`
* Row `20965` is `' refund'`
* Row `9834` is `' yes'` / Row `902` is `' no'`

Because `Qwen2.5-0.5B` and `1.5B` tie their input and output embeddings (`tie_word_embeddings=True`), the vector the model uses to *output* a word is the exact same vector it uses to *read* that word.

### How Closed-World `Constrained` & `NOUL` Decisions Work on `lm_head`

1. **Few-Shot Anchoring ([`GUIDED_ANCHOR_PROMPT` in `pada/tree_engine/constants.py#L51-L63`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L51-L63)):**
   If you feed a raw base model `"Situation: ... Question: What is the issue? Answer:"` with zero examples, the #1 token in `lm_head` is `' that'` (**62.5%** on my Stage 1 probe) or `' the'`, because the base model wants to start a full sentence (*"The issue is that..."*).
   My engine is **training-free, but few-shot anchored**: I prepend [`GUIDED_ANCHOR_PROMPT`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L51-L63), which contains 8 short, domain-disjoint examples (Astronomy and Pastry Baking: `telescope`, `star`, `gamma`, `no`, `tart`, `butter`, `chef`, `yes`). By the time the model reaches `One-word answer:`, its hidden state is locked into emitting a single standalone word.
2. **Termination-Boundary Scoring ($P(\text{stop})$ Normalization) — A Practical Example ([`score_all_constrained_fused` in `pada/tree_engine/constrained.py#L128-L223`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L128-L223)):**
   Why can't you just multiply the probabilities of a candidate word's tokens and stop there?
   Imagine two of your multiple-choice options are **`"auto"`** (1 token: `[" auto"]`) and **`"automatic"`** (2 tokens: `[" auto", "matic"]`), and the model 100% wants to answer **`"automatic"`**:
   * **Without $P(\text{stop})$ (The Bug):** To write `"automatic"`, the model *has* to start with `" auto"` (`0.80` probability), followed by `"matic"` (`0.95` probability). If you just multiply token probabilities:
     * Score for `"auto"` = **`0.80` (80%)**
     * Score for `"automatic"` = `0.80 × 0.95` = **`0.76` (76%)**
     Because multiplying by a second piece (`× 0.95`) always makes a fraction smaller, **`"auto"` beats `"automatic"` (`80% > 76%`)** even when the model is trying to spell `"automatic"`!
   * **With $P(\text{stop})$ (The Fix):** Using [`score_all_constrained_fused()` (`pada/tree_engine/constrained.py#L207-L221`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L207-L221), I ask one extra question after each option's tokens: *"What is the probability that the model puts a period `.`, comma `,`, or newline `\n` right here (`stop_boundary_mask`)?"*
     * After `" auto"`, the model does *not* want to stop (`P(stop) = 0.02`, because it wants to write `"matic"` next) $\rightarrow$ `"auto"` drops to `0.80 × 0.02` = **`1.6%`**.
     * After `" auto" + "matic"`, the word is finished (`P(stop) = 0.98`) $\rightarrow$ `"automatic"` scores `0.76 × 0.98` = **`74.5%`** and wins cleanly!
   $$\text{Score}(\text{word}) = P(\text{tokens}) \times P(\text{stop boundary} \mid \text{tokens})$$
   *(Note: Multiplying by $P(\text{stop})$ corrects tokenization length and prefix bias; however, unlike Track 1, I have not yet fitted post-hoc temperature scaling or measured Expected Calibration Error [ECE] on Track 2, so these normalized shares should be interpreted as relative model scores rather than verified calibrated probabilities).*
3. **2D Out-of-Distribution (OOD) Guard & [`QUESTION_STOPWORDS` (`pada/tree_engine/constants.py#L39-L48`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L39-L48):**
   What if someone asks a nonsense trick question on a hospital story, like *"What did the **astronaut** repair on the **lunar** **spaceship**?"* or forces a softmax between `["mortgage", "parliament", "spreadsheet"]`?
   * First, [`finalize_constrained_dict()` in `pada/tree_engine/constrained.py#L22-L54`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L22-L54) tracks **Raw Probability Mass** ($\sum P(\text{candidates})$ before normalizing): on the `"mortgage"` test, raw mass drops to **`7.38e-08`**.
   * Second, [`compute_question_premise_ll()` in `pada/tree_engine/tree_builder.py#L66-L82`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L66-L82) checks how "surprised" the model is by the question itself (**Question Premise Log-Likelihood**). If you averaged across all words in *"What did the astronaut repair on the lunar spaceship?"*, 6 of the 9 words (`What`, `did`, `the`, `on`...) are everyday question words that appear in every English question and would hide the surprise! So [`find_question_premise_offsets()` in `pada/tree_engine/lexicon.py#L146-L162`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/lexicon.py#L146-L162) crosses out [`QUESTION_STOPWORDS`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L39-L48) (`GRAMMAR_GLUE` + `"what"`, `"which"`, `"did"`, `"does"`...) and only measures surprise on the **meaningful topic words** (`"astronaut"`, `"repair"`, `"lunar"`, `"spaceship"`), immediately flagging `is_ood = True`.

> **✅ Milestone Reached — The Closed-World Jev System Is Complete:**
> At this point, I had a complete, domain-general implementation of the Jev paradigm: shared context trunk, isolated Tree Attention branches, $P(\text{stop})$-normalized `choice` over any candidate list, and `noul` (`yes`/`no`) boolean decisions.

---

## 7. Going One Step Beyond Jev: Introducing `Nameless` (`Open Mode`)

Here is the question that pushed me past a standard Jev reconstruction:

**What happens when you don't have a list of candidate options?**

In TypeSafe AI's Jev—and in every open-source Jev clone (`Kev`, `SemIf`, `sarvam-jev`, `Nimble`)—every classification question is **closed-world**: you *must* pass the possible answers (`["Harvard", "Oxford", "MIT"]` or `["piano", "cello", "violin"]`) inside the request. If you don't already know the candidate entities inside a document, a closed-world pointer head cannot extract them.

I wanted to ask open-ended questions like:
* *"Which university did Amara graduate from?"* (with **no** list of universities)
* *"Which musical instrument does Daniel enjoy playing?"* (with **no** list of instruments)
* *"What is the primary issue in this ticket?"* (with **no** predefined taxonomy)

—and evaluate those **Open Mode** questions **inside the exact same Tree Attention pass** right alongside closed-world `CONSTRAINED` and `NOUL` questions.

I call this extended capability **`Nameless` (`Open Mode` in [`pada/tree_engine/open_mode.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py))**: an extension of the Jev architecture from closed-world option scoring to **open-vocabulary single-word extraction**.

---

### How `Nameless` (`Open Mode`) Works Today

#### 1. Filtering the 151,936-Token Vocabulary Down to Real Content Words ([`build_vocab_masks` in `pada/tree_engine/lexicon.py#L94-L143`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/lexicon.py#L94-L143))
When the model reaches `One-word answer:` at the end of an Open Mode branch, there is no candidate list—there are all **151,936 tokens** in `lm_head`!
Most of those 151,936 tokens are punctuation, numbers, code symbols, subword fragments, or conversational filler words (`"the"`, `"it"`, `"is"`, `"to"`).
In [`build_vocab_masks()` (`pada/tree_engine/lexicon.py#L94-L143`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/lexicon.py#L94-L143), I precompute a boolean mask called **`content_word_mask` (`50,780` tokens)** that keeps only space-prefixed alphabetic words while blocking **35 [`GRAMMAR_GLUE` filler words (`pada/tree_engine/constants.py#L31-L37`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L31-L37)**:
```python
GRAMMAR_GLUE = {
    "a", "an", "the", "this", "that", "these", "those", "i",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "to", "of", "in", "for", "with", "on", "at", "from", "by",
    "into", "onto", "about", "through", "under", "over",
    "and", "or", "as", "than",
}
```
Blocking `GRAMMAR_GLUE` prevents the model from starting a chatty sentence (*"The..."*, *"It..."*, *"In..."*) and forces 100% of its probability mass onto actual content nouns, verbs, and adjectives.

#### 2. How I Caught and Fixed the `'cello'` Subword Bug ([`is_bound_bpe_prefix` in `pada/tree_engine/lexicon.py#L69-L80`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/lexicon.py#L69-L80))
In Open Mode, most answers (`"harvard"`, `"daniel"`, `"toronto"`, `"oxford"`) are single tokens in Qwen's dictionary—so they are predicted immediately in **Pass 1**!
However, some words split into multiple BPE subword pieces: for example, `"locust"` is `['Ġloc', 'ust']`, and when I first ran my reading comprehension story ([`scripts/test_story_comprehension.py`](https://github.com/Akintunde102/pada/blob/main/scripts/test_story_comprehension.py)) asking *"Which musical instrument does Daniel enjoy playing?"*, `Qwen2.5-1.5B` output **`"cell"`** instead of **`"cello"`**!

Here is why `"cello"` failed originally, and how I fixed it:
1. **Why `"cello"` failed:** Qwen's dictionary doesn't have `"cello"` as one token—it splits `" cello"` into two pieces:
   ```python
   >>> tokenizer.convert_ids_to_tokens(tokenizer.encode(" cello"))
   ['Ġc', 'ello']  # Token 272 (" c") + Token 4791 ("ello")
   ```
   Originally, my code had a rule: *"Ignore all 1-letter pieces (`len(clean) >= 2`), because I only want real words, not single letters."* That rule blocked **`" c"`** (`272`), forcing the model to fall back to **`" cell"`** (`2779`).
2. **Allowing 1-letter word-starters (`len >= 1`) as Bound Prefixes:** To fix `"cello"`, I updated [`pada/tree_engine/lexicon.py#L121-L125`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/lexicon.py#L121-L125) to allow 1-letter pieces (`len(clean) >= 1`), **but** marked all 1-letter pieces in **`bound_prefix_mask` (`538` bound stems)** ([`pada/tree_engine/lexicon.py#L72-L73`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/lexicon.py#L72-L73)) with `forbid_bound_stop = True`. That means if the model picks **`" c"`**, it is *forbidden* from stopping at `"c"` and is *forced* to attach its second piece **`"ello"`** $\rightarrow$ **`"cello"` (94.7% score)**!
3. **Why I added `"i"` to `GRAMMAR_GLUE`:** In English, there are two valid 1-letter words that start sentences: **`"a"`** and **`"I"`**. When I allowed 1-letter pieces (`len >= 1`), `"a"` was already blocked in `GRAMMAR_GLUE`, and adding **`"i"`** to [`GRAMMAR_GLUE` (`pada/tree_engine/constants.py#L32`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L32) ensured the model couldn't use `"I"` or `"a"` as a 1-letter filler word while still allowing 1-letter BPE word-starters like `" c"` (`"c" + "ello"`).

#### 3. Where the Engine Is Today vs. The Ultimate Goal: Single-Pass Multi-Token Extraction

Here is how Open Mode executes today in [`complete_all_open_fused()` (`pada/tree_engine/open_mode.py#L193-L391`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py#L193-L391), and where this architecture is headed next:

```
WHERE THE ENGINE IS TODAY (1 Tree Pass + 1 Fused Stage-2 KV Continuation Pass):
┌─────────────────────────────────────────────────────────────────────────┐
│ PASS 1 (pack_guided_tree in pada/tree_engine/tree_builder.py#L11):      │
│ Encodes Shared Story + All Q Branches (Open, Constrained, NOUL) at once │
│  ├─► Single-token Open words ("harvard", "toronto") are ready right now!│
│  └─► Saves tree_kv (batch_size = 1)                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ PASS 2 (complete_all_open_fused & score_all_constrained_fused):         │
│ Packs multi-token stems ('Ġc' -> 'ello') & P(stop) checks across ALL    │
│ questions directly onto tree_kv without re-encoding the story!          │
└─────────────────────────────────────────────────────────────────────────┘
```

* **Where the engine is today:**
  * Every single-token Open Mode answer is already predicted in **Pass 1**.
  * To verify $P(\text{stop})$ and complete multi-token BPE words like `'Ġc' + 'ello'` (`"cello"`) or `'Ġloc' + 'ust'` (`"locust"`), my engine runs **[`complete_all_open_fused()` in `pada/tree_engine/open_mode.py#L193-L391`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py#L193-L391)**—a fused Stage-2 continuation pass directly on top of Pass 1's `tree_kv` (`batch_size = 1`, zero KV duplication).
* **The Ultimate Goal (True 1-Pass Multi-Token Open Mode — And Why It Is Not Far-Fetched):**
  Can Pass 2 be eliminated entirely so that even multi-token words like `'Ġc' + 'ello'` are extracted in **a single forward pass (Pass 1)**?
  **Yes—and the building blocks already exist in recent ML research:**
  1. **Parallel Multi-Token Prediction (MTP) / Medusa Subword Heads:** In standard LLMs, the final hidden state $h_{\text{readout}}$ only projects through `lm_head` to predict token $t_1$. However, as demonstrated by **Medusa** (Cai et al., 2024) and **DeepSeek-V3's Multi-Token Prediction (MTP)**, attaching 2 or 3 lightweight parallel projection heads ($W^{(1)}, W^{(2)}, W^{(\text{stop})}$) to $h_{\text{readout}}$ allows a **single forward pass** to simultaneously predict Subword 1 (`'Ġc'`), Subword 2 (`'ello'`), and $P(\text{stop})$ from the exact same `<|readout|>` position!
  2. **Pooled Whole-Word Lexicon Readout:** Even without training Medusa heads, you can construct a static whole-word readout matrix $W_{\text{lexicon}} \in \mathbb{R}^{W \times d}$ where multi-token English words (`"cello"`, `"cardiology"`, `"locust"`) are represented by composing their constituent rows from `lm_head.weight` (or probing the last 2 transformer layers jointly), turning multi-token word scoring into a single matrix multiplication at the end of Pass 1.

In short: **Track 1 + Constrained/NOUL Mode** gives you the complete closed-world Jev system today; **`Nameless` (`Open Mode`)** takes it one step further into open-vocabulary extraction using a fused 2-stage pass; and **parallel MTP subword heads** are the natural next step to collapse multi-token open extraction into a single forward pass.

---

## 8. Profiling, Speed Benchmarks, Development Demo & Scenario Suite Results

### Benchmark 1: Stage-by-Stage Profiling & Wall-Clock Speed Comparison ([`scripts/benchmark_speed_comparison.py`](https://github.com/Akintunde102/pada/blob/main/scripts/benchmark_speed_comparison.py))

When I moved my engine from CPU onto my local laptop GPU (**NVIDIA Quadro P1000**, 4 GB VRAM, `float16`) and ran [`scripts/benchmark_speed_comparison.py`](https://github.com/Akintunde102/pada/blob/main/scripts/benchmark_speed_comparison.py), profiling each internal stage taught me a critical lesson:
* **Pass 1 (`Tree Prefill` for all 12 questions)** took only **`589.6 ms`** (`49 ms/question`), and **Pass 2a ([`score_all_constrained_fused`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L128) for all 7 Constrained/NOUL questions)** took only **`354.3 ms`**.
* Yet my first end-to-end GPU run took **`4,674.2 ms`**! Why? Because while Constrained Mode was fused into a single Stage-2 pass on `tree_kv`, my initial Open Mode ([`complete_batch_branched` in `pada/tree_engine/open_mode.py#L19`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py#L19)) was being called inside a Python `for` loop **once per Open question** (`5` separate calls, duplicating a `batch_size = 10` KV cache each time).
* **The Fused Stage-2 Open Mode Fix ([`complete_all_open_fused` in `pada/tree_engine/open_mode.py#L193-L391`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py#L193-L391)):** Writing `complete_all_open_fused()` to pack all Open Mode subword continuations across all 5 Open questions directly onto Pass 1's `tree_kv` (`batch_size = 1`, zero KV duplication) immediately cut full end-to-end GPU latency by **2.5×**—from **`4,674.2 ms` down to `1,892.7 ms`**!

#### Stage-by-Stage Internal GPU Profile ($Q = 12$ Mixed Questions on `Quadro P1000`, `Qwen2.5-0.5B`)
| Pipeline Stage ([`TreeEngine.predict()`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/engine.py#L112)) | What Happens Inside This Stage | GPU Time (`ms`) | Share of Total |
| :--- | :--- | :--- | :--- |
| **Pass 1 ([`pack_guided_tree`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11))** | Encodes shared story + all 12 question branches in 1 4D Tree Attention pass; extracts single-token logits & `tree_kv` | **589.6 ms** (`49 ms/q`) | **29.4%** |
| **Pass 2a ([`score_all_constrained_fused`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L128))** | Scores all candidate tokens + $P(\text{stop})$ across all 7 Constrained & NOUL questions in 1 fused pass on `tree_kv` | **354.3 ms** (`51 ms/q`) | **17.7%** |
| **Pass 2b ([`complete_all_open_fused`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py#L193))** | Completes multi-token BPE subwords (`'Ġc' + 'ello'`) + $P(\text{stop})$ across all 5 Open Mode questions on `tree_kv` | **1,062.3 ms** (`212 ms/q`) | **52.9%** |
| **Full `TreeEngine.predict()` Total** | **All 12 questions (5 Open + 3 Choice + 4 NOUL) + 2D OOD scoring** | **2,006.2 ms** (`167 ms/q`) | **100.0%** |

---

#### End-to-End Speed Comparison: Packed Tree vs. Sequential Calls vs. JSON `.generate()` ([`scripts/benchmark_speed_comparison.py#L34-L210`](https://github.com/Akintunde102/pada/blob/main/scripts/benchmark_speed_comparison.py#L34-L210))
To make the comparison crystal clear, [`scripts/benchmark_speed_comparison.py`](https://github.com/Akintunde102/pada/blob/main/scripts/benchmark_speed_comparison.py) measures four columns across $Q = 4, 8,$ and $12$ questions:
* **1a. Tree Prefill ([`pack_guided_tree`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11)):** Pass 1 shared-trunk forward pass (what a 1-pass readout engine takes).
* **1b. Full [`TreeEngine.predict()`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/engine.py#L112):** Complete end-to-end pipeline (Pass 1 + Pass 2a Fused Constrained + Pass 2b Fused Open Mode + $P(\text{stop})$ + OOD).
* **2. $Q$ Separate Sequential Calls:** Standard approach calling `model(story + question_i)` $Q$ separate times (re-reading the story $Q$ times).
* **3. Autoregressive JSON `.generate()`:** Standard chat approach generating `{"key": "val"}` token-by-token (`6` tokens per question).

##### A. Local Laptop GPU (`NVIDIA Quadro P1000`, 4 GB VRAM, `float16`)
| Model & Question Count | 1a. Tree Prefill | 1b. Full `TreeEngine` | 2. $Q$ Sequential Calls | 3. JSON `.generate()` | Full Engine Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`0.5B` ($Q = 4$)** | **414.2 ms** (`104 ms/q`) | **1,730.9 ms** (`433 ms/q`) | 1,860.3 ms (`465 ms/q`) | 4,522.8 ms (`24 tok`) | **1.07× vs. Seq** (`2.61× vs. JSON`) |
| **`0.5B` ($Q = 8$)** | **481.4 ms** (`60 ms/q`) | **1,842.0 ms** (`230 ms/q`) | 3,526.1 ms (`441 ms/q`) | 8,726.1 ms (`48 tok`) | **1.91× vs. Seq** (`4.74× vs. JSON`) |
| **`0.5B` ($Q = 12$)** | **581.2 ms** (`48 ms/q`) | **1,892.7 ms** (`158 ms/q`) | 4,726.7 ms (`394 ms/q`) | 12,459.7 ms (`72 tok`) | **2.50× vs. Seq** (**6.58× vs. JSON**) |
| **`1.5B` ($Q = 12$)** | **4,971.3 ms** (`414 ms/q`) | **5,464.3 ms** (`455 ms/q`) | 28,938.3 ms (`2,412 ms/q`) | 17,012.7 ms (`72 tok`) | **5.30× vs. Seq** (**3.11× vs. JSON**) |

##### B. CPU (`float32`, `Qwen/Qwen2.5-0.5B`)
| Question Count ($Q$) | 1a. Tree Prefill | 1b. Full `TreeEngine` | 2. $Q$ Sequential Calls | 3. JSON `.generate()` | Full Engine Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$Q = 4$** | **3,024.2 ms** (`756 ms/q`) | **5,714.6 ms** (`1,429 ms/q`) | 14,006.2 ms (`3,502 ms/q`) | 5,244.7 ms (`24 tok`) | **2.45× vs. Seq** (`0.92× vs. JSON`) |
| **$Q = 8$** | **2,986.9 ms** (`373 ms/q`) | **5,770.5 ms** (`721 ms/q`) | 19,988.3 ms (`2,499 ms/q`) | 9,956.5 ms (`48 tok`) | **3.46× vs. Seq** (`1.73× vs. JSON`) |
| **$Q = 12$** | **3,620.4 ms** (`302 ms/q`) | **6,714.2 ms** (`560 ms/q`) | 26,225.8 ms (`2,185 ms/q`) | 14,354.6 ms (`72 tok`) | **3.91× vs. Seq** (**2.14× vs. JSON**) |

* **Key Takeaways from the Numbers:**
  1. **GPU vs. CPU:** On the 12-question story test with `Qwen2.5-1.5B`, moving from CPU (`39,652 ms`) to my local `Quadro P1000` GPU (`3,899 ms` on [`scripts/test_story_comprehension.py`](https://github.com/Akintunde102/pada/blob/main/scripts/test_story_comprehension.py), or `5,464 ms` inside the isolated benchmark loop) delivers an **~7×–10× speedup** while preserving the same **11 confident passes + 1 borderline pass** (`52.1%` in `fp32`, `51.9%` in `fp16`).
  2. **Tree Attention vs. JSON Generation:** On GPU at $Q = 12$ (`0.5B`), `1a. Tree Prefill` (`581.2 ms`) is **21.4× faster** than autoregressive JSON `.generate()` (`12,459.7 ms`), and the full multi-token `TreeEngine.predict()` (`1,892.7 ms`) is **6.58× faster** than JSON generation.
  3. **Why Modern Tensor-Core GPUs (T4, L4, RTX 4090, A100) Should Scale Even Further:** The `Quadro P1000` is a 4 GB mobile Pascal GPU (`sm_61`, 2017) with no Tensor Cores. Because Tree Attention converts $Q$ separate queries and 72 sequential decode steps into dense matrix-matrix (`GEMM`) operations, I expect modern Tensor-Core GPUs to widen the speedup over memory-bandwidth-bound `.generate()` even further—though I have only benchmarked my local CPU and `Quadro P1000` so far.

---

### Demo 2: Literature Reading Comprehension Walkthrough ([`scripts/test_story_comprehension.py`](https://github.com/Akintunde102/pada/blob/main/scripts/test_story_comprehension.py))
*(Methodology note: This 12-question story is a **development demo**, not a held-out benchmark—I used this story to diagnose the `'Ġc' + 'ello'` prefix bug and added the two boolean `yes`/`no` examples to [`GUIDED_ANCHOR_PROMPT` (`pada/tree_engine/constants.py#L57-L62`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L57-L62) after observing `0.5B` fail on boolean questions).*

> **Story:** *"Amara Okafor graduated from Harvard with a degree in architecture before marrying Daniel Mensah in Lagos. Last autumn, the couple moved to Toronto so Amara could lead a museum renovation project. Even though Daniel enjoys playing the cello in his free time, he has been furious all week because the airline lost his luggage during the flight. Meanwhile, Amara's younger sister Chidinma still attends Oxford, where she studies robotics and volunteers at a local hospital on weekends."*

All **12 questions** run together on `Qwen/Qwen2.5-1.5B`, achieving **11 confident passes and 1 borderline pass (`52.1%`)**:

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
| `noul_sister_in_toronto` | Does Chidinma attend school in Toronto? | **NOUL (Y/N)** | `no` | `yes` (71.7%) | `no` (**52.1%**) | ⚠️ **Borderline** *(near 50/50 coin flip)* |
| **Summary** | | | | **6 / 12** | **11 / 12 confident (+1 borderline)** | |

---

### Evaluation 3: The 11-Scenario Multi-Domain Suite ([`pada/tree_engine/suites.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/suites.py))
In [`pada/tree_engine/suites.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/suites.py), I evaluated both **`Qwen2.5-0.5B`** and **`Qwen2.5-1.5B`** across 11 scenarios: 7 Primary Dev scenarios (`23` accuracy-scored questions + `2` unscored `expected=None` OOD negative-control questions in Scenario 7 = `25` questions) and 4 Secondary Domain scenarios (`12` questions across Hospitality, Port Logistics, University, and Agriculture), totaling **`35` accuracy-scored questions** (`37` total questions evaluated including the 2 OOD negative controls).

I also tagged each of the 35 scored questions along two axes:
* **By Mode:** `Open Mode` (`11` questions) vs. `Constrained Mode` (`24` questions).
* **By Reasoning Type:** `Extractive` (`13` questions where the answer word appears verbatim in the story, e.g. `"professor"`, `"maize"`) vs. `Inferential` (`22` questions requiring synthesis or domain reasoning where the answer word is *not* in the story, e.g. `"cardiology"`, `"maritime"`, `"indemnification"`). Note that Constrained Mode has a lower score than Open Mode on `0.5B` because **17 of the 24 Constrained questions are hard inferential traps** (`wire` vs. `phishing`, `maritime` vs. `rail`, `sim` vs. `SIM`), whereas most Open Mode questions are extractive.

| Suite / Category (`35` Scored + `2` OOD Controls = `37` Total) | `Qwen2.5-0.5B` | `Qwen2.5-1.5B` | Notes |
| :--- | :--- | :--- | :--- |
| **Secondary Domain Suite (`12` questions)** | **83.3%** (`10 / 12`) | **83.3%** (`10 / 12`) | Hospitality, Port Logistics, University, Agriculture |
| **Primary Dev Suite (`23` scored questions)** | **65.2%** (`15 / 23`) | **69.6%** (`16 / 23`) | Adversarial prefix-collision & multi-token edge cases |
| **Constrained Mode (`24` questions)** | **66.7%** (`16 / 24`) | **75.0%** (`18 / 24`) | 17 of 24 require inferential reasoning beyond surface text |
| **Open Mode (`11` questions)** | **81.8%** (`9 / 11`) | **72.7%** (`8 / 11`) | Misses on `0.5B` were synonyms (`urgent`/`critical`, `broken`/`damaged`) |
| **Extractive Subset (`13` of 35 questions)** | **84.6%** (`11 / 13`) | **69.2%** (`9 / 13`) | Answer word appears verbatim in the situation text |
| **Inferential Subset (`22` of 35 questions)** | **63.6%** (`14 / 22`) | **77.3%** (`17 / 22`) | Answer requires reasoning (`1.5B` jumps +13.7% here) |
| **OOD Assertions (`3` checks in Scenario 7)** | **2 / 3 Passed** | **3 / 3 Passed** | `1.5B` passes all 3 gross & hard-negative OOD checks |

---

## 9. How to Run and Test Everything Yourself

The full codebase ([`github.com/Akintunde102/pada`](https://github.com/Akintunde102/pada)—including Track 1, Track 2, speed benchmarks, training scripts, and all 33 unit tests) is open-source under the **[MIT License](https://github.com/Akintunde102/pada/blob/main/LICENSE)**:

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

pip install torch transformers accelerate peft fastapi uvicorn nltk datasets
```

### Way 1: Run the Speed Comparison & Stage Profiler Benchmark ([`scripts/benchmark_speed_comparison.py`](https://github.com/Akintunde102/pada/blob/main/scripts/benchmark_speed_comparison.py))
```bash
python scripts/benchmark_speed_comparison.py --model-name Qwen/Qwen2.5-0.5B
```

### Way 2: Run the 12-Question Story Comprehension Demo ([`scripts/test_story_comprehension.py`](https://github.com/Akintunde102/pada/blob/main/scripts/test_story_comprehension.py))
```bash
python scripts/test_story_comprehension.py --model-name Qwen/Qwen2.5-1.5B
```

### Way 3: Run the 11-Scenario Multi-Domain Suite & Equivalence Proofs ([`scripts/run_tree_engine.py`](https://github.com/Akintunde102/pada/blob/main/scripts/run_tree_engine.py))
```bash
# Run with mathematical equivalence verification (checks max |tree_logits - seq_logits| < 1e-3):
python scripts/run_tree_engine.py --model-name Qwen/Qwen2.5-0.5B

# Or run on 1.5B (skipping verification to jump straight to the 11 scenarios):
python scripts/run_tree_engine.py --model-name Qwen/Qwen2.5-1.5B --skip-verify
```

### Way 4: Train or Launch the Track 1 FastAPI Server & Web Studio ([`scripts/train_colab_1_5b.py`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py) & [`scripts/server.py`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py))
*(Note: Because large model checkpoints in `checkpoints/` are gitignored, running [`scripts/server.py`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py) on a fresh clone will automatically fall back to base `Qwen2.5-1.5B` weights unless you first run [`scripts/train_colab_1_5b.py`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py) to generate `checkpoints/jev_1_5b_lora_adapter` and `checkpoints/jev_1_5b_heads_and_T.pt`).*
```bash
# Optional: train the Stage A + Stage B LoRA adapter & readout heads
python scripts/train_colab_1_5b.py

# Start the FastAPI server & Studio UI on http://127.0.0.1:8000
python scripts/server.py
```

### Way 5: Run the Unit Test Suites (33 Tests: 12 Schema in [`tests/test_schema.py`](https://github.com/Akintunde102/pada/blob/main/tests/test_schema.py) + 21 Tree Engine in [`tests/`](https://github.com/Akintunde102/pada/tree/main/tests))
```bash
python tests/test_schema.py
python -m unittest discover -s tests -p "test_tree_*.py"
```

---

## 10. Honest Limitations & What's Left to Be Done

1. **No ECE / Reliability Diagram Calibration Yet on Track 2:**
   * Track 1 trains readout heads with Brier + Cross-Entropy loss and applies temperature scalars ($T$). Track 2 uses raw `lm_head` logits normalized by $P(\text{stop})$. While $P(\text{stop})$ removes tokenization length bias, I have **not** yet measured Expected Calibration Error (ECE) or fitted temperature scaling on a large labeled validation set for Track 2.
2. **OOD Raw Mass Depends on Candidate Token Length:**
   * In Track 2, my `is_ood` flag ([`pada/tree_engine/constrained.py#L38-L41`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L38-L41)) uses a raw probability mass threshold calibrated on 3-to-4-candidate examples. Because joint probability mass naturally decays for multi-token candidates, production OOD detection requires calibrating thresholds conditioned on candidate token length and option count.
3. **Single-Word Output Boundary:**
   * Track 2's Open Mode ([`pada/tree_engine/open_mode.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py)) is designed for **single-word** answers (`"Harvard"`, `"cardiology"`, `"locust"`). Multi-word entities (*"New York City"*) stop at the first complete word in Open Mode.
4. **Token Casing Sensitivity in Constrained Candidates:**
   * In Scenario 2 (SIM-swapping fraud in [`pada/tree_engine/suites.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/suites.py)), my candidate list specified lowercase `"sim"`, while the model placed 32.1% of its unconstrained probability mass on uppercase `"SIM"`. Marginalizing over lowercase, Title-case, and UPPERCASE BPE surface forms for each candidate is an easy improvement left to implement.
5. **Small Dense Models (`0.5B` / `1.5B`) vs. Larger / MoE Models:**
   * On indirect multi-hop negation (*"Chidinma attends Oxford... Does Chidinma attend school in Toronto?"*), `0.5B` failed (`71.7% yes`) and `1.5B` was borderline (`52.1% no`). I hypothesize that scaling to `Qwen2.5-7B` or an open Sparse MoE like `Qwen3-30B-A3B` or `DeepSeek-V3` will improve multi-hop boolean reasoning, though I have only tested `0.5B` and `1.5B` so far.
6. **Custom FlashInfer / FlexAttention Kernels:**
   * [`pack_guided_tree()` in `pada/tree_engine/tree_builder.py#L58-L59`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L58-L59) currently materializes dense 4D masks (`[1, 1, N, N]`) for PyTorch SDPA. Replacing this with **PyTorch FlexAttention `BlockMask`** or **FlashInfer Cascade Attention** will skip masked branch blocks at the CUDA kernel level.

---

## 11. References & Further Reading

* **TypeSafe AI's Jev & `archerhume`'s Probe:**
  * [InfoQ: TypeSafe AI Releases Jev](https://infoq.com/news/2026/10/typesafe-ai-jev-released)
  * [The Register: TypeSafe AI Debuts Model for Machines](https://www.theregister.com/a/5296711)
  * [VKTR: ChatGPT Co-Creator Launches TypeSafe AI with Jev](https://www.vktr.com/ai-platforms/chatgpt-cocreator-launches-typesafe-ai-with-jev/)
  * [TechForge: Early Access and Funding Coverage](https://www.artificialintelligence-news.com/news/chatgpt-pioneer-launches-jev-model-for-programmatic-logic/)
  * [The New Stack: TypeSafe's Jev System One Model](https://thenewstack.io/typesafe-jev-system-one/)
  * [BestHub Republished Summary of `archerhume`'s 10,000-Call Jev Probe](https://www.besthub.dev/articles/reverse-engineering-jev-10k-api-calls-expose-closed-source-model-architecture-27085f944485)
  * [Sébastien Dubois: Reinforcement Learning for Calibrated Decisions (RLCD) & Rajesh Beri's Synthetic Ticket Evaluation](https://www.dsebastien.net/reinforcement-learning-for-calibrated-decisions-rlcd/)
  * [Towards AI: How Does Jev Work? A Technical Reconstruction from 5 Public Clues](https://pub.towardsai.net/how-does-jev-work-a-technical-reconstruction-from-5-public-clues-c32b29dcb3d5)
* **Open-Source Jev Reconstructions & Comparisons:**
  * [Jared Palmer: `jaredpalmer/kev` (Open-Source Decision Model)](https://github.com/jaredpalmer/kev) & [The Menon Lab Analysis of Kev](https://themenonlab.blog/blog/kev-open-source-decision-model)
  * [TheoLeeCJ: `TheoLeeCJ/SemIf-OpenJev` (Semantic Ifs / OpenJev)](https://github.com/TheoLeeCJ/SemIf-OpenJev)
  * [Sagar Tamang: `SAGAR-TAMANG/sarvam-jev` Wiki](https://github.com/SAGAR-TAMANG/sarvam-jev/wiki)
  * [Kye Gomez: `kyegomez/open-jev`](https://github.com/kyegomez/open-jev)
  * [Lilting: Jev Clones Architecture Comparison (Kev, SemIf, Laya, Nimble, jevos)](https://lilting.ch/en/articles/jev-clones-architecture-comparison)
  * [Analytics Vidhya: Jev Projects on GitHub](https://www.analyticsvidhya.com/blog/2026/10/jev-projects-on-github/)
* **Prior Art on Verbalizers & Log-Likelihood Scoring:**
  * Schick & Schütze (2021), *"Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference"* (EACL 2021).
  * Gao et al., *"A Framework for Few-Shot Language Model Evaluation"* (`lm-evaluation-harness`).
* **Calibration & Latent Knowledge in LLMs:**
  * Kadavath et al. (Anthropic), *"Language Models (Mostly) Know What They Know"* (arXiv:2207.05221).
  * Burns et al., *"Discovering Latent Knowledge in Language Models Without Supervision"* (ICLR 2023).
  * Glenn W. Brier (1950), *"Verification of forecasts expressed in terms of probability"*.
* **Tree Attention, Shared-Prefix Systems & Multi-Token Prediction:**
  * Juravsky et al. (2024), *"Hydragen: High-Throughput LLM Inference with Shared Prefixes"*.
  * Yao et al. (2024), *"DeFT: Decoding with Flash Tree-Attention for Efficient Tree-Structured LLM Inference"*.
  * Zheng et al. (2024), *"SGLang: Efficient Execution of Structured Language Model Programs"* (RadixAttention).
  * Cai et al. (2024), *"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"*.
  * Gloeckle et al. (2024), *"Better & Faster Large Language Models via Multi-token Prediction"* (ICML 2024) & DeepSeek-AI (2024), *"DeepSeek-V3 Technical Report"* (Multi-Token Prediction / MTP).
  * PyTorch Blog, *"FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention"*.
