# Answering Multiple Questions in One LLM Pass: How I Built an Open-Source Jev Engine in PyTorch

*Using Tree Attention and the model's built-in vocabulary table (`lm_head`) to get multiple-choice, yes/no, and single-word answers without generating text token-by-token.*

**Code repository (MIT License):** [github.com/Akintunde102/pada](https://github.com/Akintunde102/pada)  
Every file, class, function, and benchmark mentioned below links directly to the exact file and line numbers in that repo.

> **Note on how this post is organized:** I wrote this so someone new to machine learning can read it top-to-bottom without getting lost in jargon. If you already work with Transformers and PyTorch, each technical section starts with a short **ML Summary** block containing the tensor shapes, equations, and code paths so you can skip the plain-English analogy right below it.

---

## 1. Why I built this

On September 15, 2026, TypeSafe AI (founded by Diogo Almeida, Erik Gafni, and Sasha Sheng) launched **Jev** in early access. They described it as a "System One" model built for fast, structured decisions rather than chat, trained with **RLCD (Reinforcement Learning for Calibrated Decisions)**.

Not long after launch, an independent researcher going by **`archerhume`** ran 10,000 API calls against Jev over two weeks to see how it behaved under the hood ([BestHub summary of the probe](https://www.besthub.dev/articles/reverse-engineering-jev-10k-api-calls-expose-closed-source-model-architecture-27085f944485)). Two observations from that probe stood out to me:
1. A 200-option multiple-choice query returned `output_tokens=1911` in the usage field at the exact same latency as a 2-option query. Since TypeSafe's docs state that decisions are billed on input tokens only, `output_tokens` is just a synthetic usage counter. The model is not generating tokens in a loop.
2. When you pass one piece of context (like a customer support ticket) and ask several questions about it, the context prefix is encoded once and shared across the questions, which are evaluated independently.

That got me thinking about how most production software still uses LLMs for classification, ticket routing, fraud checks, and entity extraction. Usually, people prompt a chat model to output JSON (`{"department": "payments", "urgent": true}`) and wait while `.generate()` decodes it one token at a time. Even with a KV cache, every single generated token forces the GPU to stream billions of model weights from memory into the compute cores.

You can avoid full JSON decoding in standard inference engines (like vLLM or SGLang) by setting `max_tokens=1` with `logprobs=True` and turning on prefix caching. That gets you pretty far for simple single-token labels, but I kept running into three cases where `max_tokens=1` breaks down:
1. Answering $Q$ different questions about one document still requires dispatching $Q$ separate requests or batch rows instead of packing isolated question branches into one forward pass.
2. If a candidate word splits into multiple BPE tokens (like `' cardiology'` $\rightarrow$ `['Ġcard', 'iology']`), `max_tokens=1` only scores the first piece `'Ġcard'` and cannot check whether the word actually ends where it should ($P(\text{stop} \mid \text{candidate})$).
3. If you don't have a fixed list of multiple-choice options and just want an open-ended single-word answer, `max_tokens=1` gets derailed by conversational starters (`' that'`, `' the'`) or half-words (`'Ġloc'`, `'Ġc'`).

There is also the calibration side. Asking a chat model to print `"90% confident"` as text does not guarantee that 90% of its predictions at that confidence level are actually right. TypeSafe uses RLCD to optimize the output probability distributions directly across three primitives: `choice` (picking from options), `score` (numeric rubrics), and `noul` (binary yes/no). Independent evaluations of Jev's calibration vary depending on the dataset: `archerhume` measured an Expected Calibration Error (ECE) of `0.0313` on MMLU, while an [independent synthetic support-ticket test by Rajesh Beri, cited by Sébastien Dubois](https://www.dsebastien.net/reinforcement-learning-for-calibrated-decisions-rlcd/), measured an ECE of `0.107`.

### Related open-source projects and what is in `pada`

Jev's weights are proprietary, so this project is not a replica of TypeSafe's internal training pipeline. Several other developers have also built open-source Jev-style engines recently, including Jared Palmer's **[Kev](https://github.com/jaredpalmer/kev)** (`jaredpalmer/kev`, which uses a Qwen base model, a LoRA adapter, a pointer head, and shared-state attention masking; see [The Menon Lab write-up](https://themenonlab.blog/blog/kev-open-source-decision-model)), **[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev)** (`TheoLeeCJ/SemIf-OpenJev`), Sagar Tamang's **[sarvam-jev](https://github.com/SAGAR-TAMANG/sarvam-jev/wiki)** (`SAGAR-TAMANG/sarvam-jev`), Kye Gomez's **[`kyegomez/open-jev`](https://github.com/kyegomez/open-jev)**, **Laya**, **Nimble**, and **jevos** (compared in the [Lilting architecture roundup](https://lilting.ch/en/articles/jev-clones-architecture-comparison)).

In my repo ([`Akintunde102/pada`](https://github.com/Akintunde102/pada), tested on `Qwen2.5-0.5B` and `Qwen2.5-1.5B`), I ended up building two separate tracks:
* **Track 1 ([`pada/schema/`](https://github.com/Akintunde102/pada/tree/main/pada/schema) and [`scripts/server.py`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py)):** My first attempt. Like `Kev`, it adds 7 custom special tokens (`<|state|>`, `<|q|>`, `<|opt|>`, `<|readout|>`, `<|noul|>`, `<|choice|>`, `<|score|>`), strips JSON keys on the host CPU via [`HostModelAdapter` (`pada/schema/internal.py#L73`)](https://github.com/Akintunde102/pada/blob/main/pada/schema/internal.py#L73), and trains three custom readout heads ([`NoulHead`, `ChoiceHead`, and `UniversalScoreHead` in `scripts/server.py#L59-L97`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L59-L97)) using [`scripts/train_colab_1_5b.py`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py).
* **Track 2 ([`pada/tree_engine/`](https://github.com/Akintunde102/pada/tree/main/pada/tree_engine)):** What I built after Track 1 failed on out-of-domain prompts. Track 2 drops custom readout heads completely and probes the base model's existing `lm_head` ([`TreeEngine` in `pada/tree_engine/engine.py#L37`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/engine.py#L37)). It pairs a 4D Tree Attention prefill pass ([`pack_guided_tree` in `pada/tree_engine/tree_builder.py#L11`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11)) with a fused Stage-2 KV-cache continuation pass to handle closed-world `choice` and `noul` questions, plus an open-vocabulary mode I call **`Nameless` ([`pada/tree_engine/open_mode.py#L193`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py#L193))** that extracts single-word answers without a candidate list.

---

## 2. Prefill without the decode loop

> **ML Summary:** Causal transformers process all prompt tokens in parallel during the compute-bound **prefill** stage using lower-triangular attention masking. Autoregressive **decoding** is memory-bandwidth bound because it runs $O(N)$ sequential forward passes, reading the full weight matrices for one token at a time. Both tracks in `pada` skip autoregressive JSON decoding, read hidden states $h_{\text{last}} \in \mathbb{R}^d$ (or `lm_head` logits) at the branch boundary right after prefill, and run at most one short, fused KV-cached continuation pass for multi-token words.

### Plain-English intuition

Think of an inspector reading a one-page incident report with three questions at the bottom.
* **Prefill (reading the page):** As their eyes scan the page and reach the end of each question, they already know the answer.
* **Decode (typing out JSON):** In a standard chat LLM, even though the model saves its notes from reading the page in a **KV cache**, typing out `{`, `"`, `department`, `"` still requires hauling the model's multi-gigabyte weight matrices out of GPU memory into the processor cores **once for every single character or word piece**.

Both engines in `pada` stop right after the reading step. At the last token of each question, the model produces a vector of numbers (the hidden state) that already holds its decision. I read the scores directly from that vector and assemble the response dict in Python on the CPU.

```
Standard LLM (.generate):
[Prefill Prompt] ──> [Load Weights for Token 1] ──> [Load Weights for Token 2] ──> ... (Sequential Passes)

Tree Prefill Engine:
[Prefill Context + All Questions in Parallel] ──> [Extract Scores Directly] (0 JSON Tokens Decoded)
```

---

## 3. Tree Attention and restarted position IDs ([`pada/tree_engine/tree_builder.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11-L64))

> **ML Summary:** Given a shared state prefix of length $S$ (segment $0$) and $M$ question branches (segments $1 \dots M$), [`pack_guided_tree()` (`pada/tree_engine/tree_builder.py#L11-L64`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11-L64) packs all tokens into a single 1D sequence of length $N = S + \sum_{k=1}^M Q_k$ and builds a 4D additive attention mask $A \in \{0, -\infty\}^{1 \times 1 \times N \times N}$ where token $i$ attends to token $j$ iff:
> $$j \le i \quad \land \quad (\text{seg}_j = 0 \lor \text{seg}_i = \text{seg}_j)$$
> Because RoPE (Rotary Position Embeddings) depends on relative token distance, leaving sequential position IDs across packed branches would push later questions further away from the context. To keep every branch's relative distance to the context identical and make results invariant to question ordering, every branch $k > 0$ resets its position counter to start at $S$: $\text{pos}_i = S + (i - \text{start}_k)$. In `float32`, a single packed tree pass matches separate sequential forward passes to within `max |tree_logits - seq_logits| < 1e-3` (tested in [`pada/tree_engine/verification.py#L33-L93`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/verification.py#L33-L93)). Masked entries use `torch.finfo(dtype).min` ([`pada/tree_engine/tree_builder.py#L59`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L59)) so `float16` and `bfloat16` don't produce `NaN`s.

### Plain-English intuition

Suppose you have one paragraph of text and 12 questions to ask about it.
* If you make 12 separate calls to the model, it re-reads the paragraph 12 times.
* If you paste all 12 questions into a single prompt one after another, Question 2 can see Question 1. Asking *"Is this customer angry?"* in Question 1 can skew the answer to *"Which department should take this ticket?"* in Question 2. Swapping the order of the questions also changes how far away they sit from the original paragraph.

To read the paragraph once, answer all 12 questions in one forward pass, and keep the questions isolated from one another, [`pack_guided_tree()` (`pada/tree_engine/tree_builder.py#L11-L64`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11-L64) does two things:
1. **Tree Attention mask:** I put the shared story in Segment `0`, Question 1 in Segment `1`, Question 2 in Segment `2`, and so on. The attention mask allows every question to look back at Segment `0` (the story) and at its own tokens, while blocking it from seeing any other question branch.
2. **Restarted `position_ids`:** Transformers assign a position index (`0, 1, 2...`) to every token. If the shared story occupies positions `0` through `49`, Question 1 starts at `50, 51, 52...`, and Question 2 also starts at `50, 51, 52...`. As far as the model's positional math is concerned, every question sits immediately after the story.

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

### Code: Building the Tree Attention mask and restarted position IDs ([`pada/tree_engine/tree_builder.py#L11-L64`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11-L64))

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
        # Every branch restarts its position counter at S
        pos_ids.extend(range(S, S + len(b_ids)))

    N = len(all_ids)
    seg_t = torch.tensor(segments)
    rows = torch.arange(N).unsqueeze(1)
    cols = torch.arange(N).unsqueeze(0)

    # Causal (cols <= rows) AND (looking at state OR looking at same branch)
    allowed = (cols <= rows) & ((seg_t.unsqueeze(0) == 0) | (seg_t.unsqueeze(1) == seg_t.unsqueeze(0)))

    # 4D additive attention mask: 0.0 for allowed, finfo.min for blocked
    mask_4d = torch.zeros((1, 1, N, N), dtype=dtype).masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)
    return torch.tensor([all_ids]), mask_4d, torch.tensor([pos_ids])
```

---

## 4. Track 1: Custom readout heads and two-stage KV caching ([`scripts/server.py`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py))

> **ML Summary:** In [`scripts/server.py`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py), [`scripts/train_colab_1_5b.py`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py), and [`pada/schema/`](https://github.com/Akintunde102/pada/tree/main/pada/schema), Track 1 registers 7 special tokens (`<|state|>`, `<|q|>`, `<|opt|>`, `<|readout|>`, `<|noul|>`, `<|choice|>`, `<|score|>`), strips dictionary keys on the host via [`HostModelAdapter` (`pada/schema/internal.py#L73`)](https://github.com/Akintunde102/pada/blob/main/pada/schema/internal.py#L73), and attaches three small readout heads ([`scripts/server.py#L59-L97`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L59-L97)) to the final hidden states $h \in \mathbb{R}^d$. Each head returns raw logits, and [`evaluate_jev_request()` (`scripts/server.py#L216-L315`)](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L216-L315) applies temperature scaling ($1 / T$) before `sigmoid` or `softmax`:
> * **`NoulHead` ([`scripts/server.py#L59-L66`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L59-L66)):** $\text{logit} = W_{\text{noul}}\,\text{LN}(h_{\text{readout}}) \implies P(\text{True}) = \sigma(\text{logit} / T_{\text{noul}})$
> * **`ChoiceHead` ([`scripts/server.py#L69-L80`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L69-L80)):** Given $h_{\text{readout}} \in \mathbb{R}^d$ and option vectors $H_{\text{opts}} \in \mathbb{R}^{M \times d}$, $\text{logits} = \frac{(W_k\,\text{LN}(H_{\text{opts}})) \cdot (W_q\,\text{LN}(h_{\text{readout}}))}{\sqrt{d}} \implies P = \text{softmax}(\text{logits} / T_{\text{choice}})$.
> * **`UniversalScoreHead` ([`scripts/server.py#L83-L95`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L83-L95)):** $\text{logit} = W_{\text{score}}\,\text{LN}(h_{\text{readout}}) \implies s_{\text{norm}} = \sigma(\text{logit} / T_{\text{score}}) \in (0, 1)$, rescaled to $[V_{\min}, V_{\max}]$ and expanded into $K$-level ordinal probabilities via [`continuous_to_ordinal_distribution()` (`scripts/server.py#L200-L213`)](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L200-L213).

### Plain-English intuition

In Track 1 ([`scripts/server.py`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py) and [`scripts/train_colab_1_5b.py`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py)), I placed special marker tokens after each candidate option (`<|opt|>`) and at the end of each question (`<|readout|>`), then trained three small linear/attention layers on top of `Qwen2.5-1.5B` (along with a rank-8 LoRA adapter on `q_proj` and `v_proj`):
1. **`NoulHead` (yes/no):** Projects the `<|readout|>` vector to a single number and passes it through a sigmoid to get a probability between `0.0` and `1.0`.
2. **`ChoiceHead` (multiple choice):** Compares a question's `<|readout|>` vector (`[d]`) against each of its `<|opt|>` vectors (`[M, d]`) via scaled dot-product attention. Because it compares against marker positions in the prompt rather than fixed output classes, it can score 2 options or 200 options with the same weights.
3. **`UniversalScoreHead` (numeric rating):** Maps `<|readout|>` to `(0.0, 1.0)` and rescales it to whatever range the request asks for (`1 to 5`, `1 to 10`, etc.).

### Code: The three readout heads ([`scripts/server.py#L59-L97`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L59-L97))

```python
import math
import torch
import torch.nn as nn

class NoulHead(nn.Module):
    """Binary Yes/No head (returns raw logit; caller applies sigmoid(logit / T_noul))."""
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h)).squeeze(-1)


class ChoiceHead(nn.Module):
    """Pointer scorer comparing 1 question vector [d] against M option vectors [M, d].
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
    """Continuous rating head (returns raw logit; caller applies sigmoid(logit / T_score))."""
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h)).squeeze(-1)
```

In [`scripts/train_colab_1_5b.py#L171-L260`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py#L171-L260), I trained these heads with Cross-Entropy plus Brier loss $(\hat{p} - y)^2$ on 1,000 Amazon product reviews (Stage A on frozen hidden states, then Stage B joint LoRA + head tuning) and set scalar temperatures ($T_{\text{noul}}=1.50, T_{\text{choice}}=2.00, T_{\text{score}}=1.50$). In [`get_state_memory()` (`scripts/server.py#L172-L180`)](https://github.com/Akintunde102/pada/blob/main/scripts/server.py#L172-L180), I added a two-stage SHA-256 prefix KV cache so repeated queries over the same document skip re-encoding the state.

---

## 5. Where Track 1 broke

I started up [`scripts/server.py`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py) with the checkpoint trained on Amazon product reviews and tested an in-domain review first:

> **State:** *"This blender broke on the second day! Plastic gears stripped completely and customer service ignored my emails. Total waste of money."*

All three heads returned sensible numbers in a single pass:
* `sentiment` (`ChoiceHead`): `neg`: `87.2%`, `neu`: `12.8%`, `pos`: `0.04%`
* `positive` (`NoulHead`): `p_true`: `12.0%`
* `rating` (`UniversalScoreHead`): `0.58 / 4.0` (tier distribution: `[53.7%, 36.1%, 9.1%, 1.0%, 0.04%]`)

Next, I passed it an out-of-domain prompt: routing a customer support ticket about failed bank payouts.

> **State:** *"My payouts have failed three times. The bank says everything is fine. Can someone fix this?"*  
> **Options:** `payments` (*"Payout failures and payment processing"*), `account` (*"Login and account access"*), `other` (*"Something else"*)

The server returned:
* `account`: `88.99%` (wrong)
* `payments`: `5.75%`
* `other`: `5.25%`

Looking at why a 1.5B model mixed up payout failures with login access, two issues were obvious:
1. `ChoiceHead` and `NoulHead` start as randomly initialized matrices (`W_q`, `W_k`, `proj`). Because [`scripts/train_colab_1_5b.py`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py) only trained them on 1,000 product reviews, the readout projections latched onto sentiment features rather than general option matching. Fixing that would mean curating a large multi-domain dataset just to teach the custom heads what the base model already knows.
2. Even a well-trained pointer head over `<|opt|>` tokens is strictly closed-world. It can only pick from options you explicitly paste into the prompt. It can never answer *"Which university did Amara graduate from?"* unless you pass a list of universities in the request.

That failure is why I built Track 2: I wanted to see if I could run both closed-world choices and open-ended single-word extraction over the same shared Tree Attention pass using the base model's existing `lm_head`, without training any new weights.

---

## 6. Running closed-world Jev decisions on `lm_head` (`Constrained` and `NOUL`)

> **ML Summary:** Every causal LLM already has a trained readout projection: `lm_head.weight` $\in \mathbb{R}^{V \times d}$ ($V = 151,936$ in Qwen2.5). Both `Qwen2.5-0.5B` and `Qwen2.5-1.5B` tie `lm_head.weight` directly to `embed_tokens.weight` (`tie_word_embeddings = True`).
> To evaluate closed-world `choice` and binary `noul` (`yes`/`no`) questions without custom heads, [`TreeEngine.predict()` (`pada/tree_engine/engine.py#L112-L295`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/engine.py#L112-L295) runs two steps:
> * **Pass 1 ([`pack_guided_tree` in `pada/tree_engine/tree_builder.py#L11-L64`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11-L64)):** Packs [`GUIDED_ANCHOR_PROMPT` (`pada/tree_engine/constants.py#L51-L63`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L51-L63) + `Situation: <state>` (Segment $0$, length $S$) and all $M$ isolated question branches (`Question: <instr> One-word answer:`, Segments $1 \dots M$, positions restarted at $S$) into one forward pass with `use_cache=True`. At each branch's final token (`readout_idx`), `outputs.logits[0, readout_idx, :]` gives the first-token log-probabilities $\log P(t_1 \mid \text{state}, q_k)$.
> * **Pass 2 ([`score_all_constrained_fused` in `pada/tree_engine/constrained.py#L128-L223`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L128-L223)):** Packs all candidate token sequences across all constrained questions directly on top of Pass 1's `tree_kv` (`batch_size = 1`) using a `[1, 1, T, P + T]` 4D continuation mask, runs one teacher-forced forward pass to score multi-token candidates (`' cardiology'` $\rightarrow$ `['Ġcard', 'iology']`) and read $P(\text{stop} \mid \text{candidate})$, and crops `tree_kv` back to $P$ via [`crop_cache()` (`pada/tree_engine/cache_utils.py#L47-L51`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/cache_utils.py#L47-L51).

### Plain-English intuition

Before trying open-ended extraction, I first needed to make Jev's closed-world tasks (`choice` and `noul`) work across any domain without fine-tuning.

At the top of `Qwen2.5` is a built-in lookup table called `lm_head` with 151,936 rows, one for every token in its vocabulary. Common words already have their own row:
* Row `33531` is `' billing'`
* Row `33671` is `' urgent'`
* Row `20965` is `' refund'`
* Row `9834` is `' yes'` and Row `902` is `' no'`

Getting clean probabilities out of `lm_head` across single-token and multi-token options required solving three practical quirks in how base LLMs tokenize text:

#### 1. Few-shot anchoring ([`GUIDED_ANCHOR_PROMPT` in `pada/tree_engine/constants.py#L51-L63`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L51-L63))
If you pass a raw base model `"Situation: ... Question: What is the issue? Answer:"` with no examples, the top token in `lm_head` is `' that'` (62.5% in my initial probe) or `' the'`, because the model wants to write a full sentence (*"The issue is that..."*).

To stop that without fine-tuning, Track 2 is training-free but few-shot anchored. At the start of Segment 0, I prepend [`GUIDED_ANCHOR_PROMPT`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L51-L63), which has 8 short one-word Q&A examples from unrelated domains (astronomy and baking: `telescope`, `star`, `gamma`, `no`, `tart`, `butter`, `chef`, `yes`). By the time the model reaches `One-word answer:` on the real question, it expects to emit a single lowercase word followed by a period.

#### 2. Termination-boundary scoring via $P(\text{stop})$ ([`score_all_constrained_fused` in `pada/tree_engine/constrained.py#L128-L223`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L128-L223))
Why can't you just multiply the probabilities of a candidate word's tokens and stop there?

Suppose two of your multiple-choice options are `"auto"` (1 token: `[" auto"]`) and `"automatic"` (2 tokens: `[" auto", "matic"]`), and the model intends to answer `"automatic"`.
* **Without $P(\text{stop})$:** To spell `"automatic"`, the model has to start with `" auto"` (say, `0.80` probability), followed by `"matic"` (`0.95` probability). If you only multiply token probabilities:
  * Score for `"auto"` = `0.80` (80%)
  * Score for `"automatic"` = `0.80 × 0.95` = `0.76` (76%)
  Because multiplying by a second probability less than 1 always shrinks the product, `"auto"` beats `"automatic"` (`80% > 76%`) even when the model is trying to spell `"automatic"`.
* **With $P(\text{stop})$:** In [`score_all_constrained_fused()` (`pada/tree_engine/constrained.py#L207-L221`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L207-L221), I check the probability that the model emits a stop token (`.`, `,`, `\n`, space, or `<|im_end|>`, tracked in `stop_boundary_mask`) immediately after the candidate's last token:
  * Right after `" auto"`, the model does not want to stop (`P(stop) = 0.02`, because 95% of its probability mass is on `"matic"`). So `"auto"` scores `0.80 × 0.02 = 0.016` (1.6%).
  * Right after `" auto" + "matic"`, the word is complete (`P(stop) = 0.98`). So `"automatic"` scores `0.76 × 0.98 = 0.745` (74.5%).

$$\text{Score}(\text{word}) = P(\text{tokens}) \times P(\text{stop boundary} \mid \text{tokens})$$

*(Note: Multiplying by $P(\text{stop})$ fixes BPE length and prefix bias, not statistical calibration. Unlike Track 1, I have not fitted temperature scalars or measured ECE on Track 2 yet, so these normalized numbers should be read as relative scores rather than calibrated probabilities.)*

#### 3. 2D out-of-distribution (OOD) guard and [`QUESTION_STOPWORDS` (`pada/tree_engine/constants.py#L39-L48`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L39-L48)
If you force a softmax to pick between `["mortgage", "parliament", "spreadsheet"]` on a medical story, the normalized shares still add up to 100% (`mortgage: 97.7%`). And what if someone asks a nonsensical question on a hospital story, like *"What did the astronaut repair on the lunar spaceship?"*

I check two numbers before trusting a constrained winner:
* **Raw probability mass** ([`finalize_constrained_dict()` in `pada/tree_engine/constrained.py#L22-L54`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L22-L54)): The unnormalized sum $\sum P(\text{candidates})$. On the `"mortgage"` test, raw mass drops to `7.38e-08`.
* **Question premise log-likelihood** ([`compute_question_premise_ll()` in `pada/tree_engine/tree_builder.py#L66-L82`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L66-L82)): How surprised the model is by the question tokens themselves given the story. If you averaged over all 9 words in *"What did the astronaut repair on the lunar spaceship?"*, 6 of them (`What`, `did`, `the`, `on`...) are generic grammar words that appear in every English question and dilute the signal. So [`find_question_premise_offsets()` (`pada/tree_engine/lexicon.py#L146-L162`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/lexicon.py#L146-L162) strips out [`QUESTION_STOPWORDS`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L39-L48) and only averages log-likelihood across the remaining domain words (`astronaut`, `repair`, `lunar`, `spaceship`). When either metric falls below threshold, `is_ood` is set to `True`.

With those three pieces in place, the closed-world Jev primitives (`choice` and `noul` `yes`/`no`) worked across arbitrary domains with zero fine-tuning.

---

## 7. Going beyond closed choices: `Nameless` (`Open Mode`)

In TypeSafe's Jev, and in existing open-source clones (`Kev`, `SemIf`, `sarvam-jev`, `Nimble`), every classification question is closed-world: you have to supply the candidate list (`["Harvard", "Oxford", "MIT"]` or `["piano", "cello", "violin"]`) in the request. If you don't know ahead of time which entities appear in a document, a closed-world option scorer cannot extract them.

I wanted to ask questions without passing any options at all:
* *"Which university did Amara graduate from?"*
* *"Which musical instrument does Daniel enjoy playing?"*
* *"What is the primary issue in this ticket?"*

and run those open-ended questions inside the exact same Tree Attention pass alongside closed-world `CONSTRAINED` and `NOUL` questions. That open-vocabulary extension is what I call **`Nameless` (`Open Mode` in [`pada/tree_engine/open_mode.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py))**.

### How `Nameless` (`Open Mode`) works

#### 1. Filtering 151,936 tokens down to content words ([`build_vocab_masks` in `pada/tree_engine/lexicon.py#L94-L143`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/lexicon.py#L94-L143))
At the end of an Open Mode branch, there is no candidate list to score. Instead, you get logits across all 151,936 tokens in `lm_head`. Most of those tokens are punctuation, numbers, code symbols, word fragments without a leading space, or common grammar words (`"the"`, `"it"`, `"is"`, `"to"`).

At startup, [`build_vocab_masks()` (`pada/tree_engine/lexicon.py#L94-L143`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/lexicon.py#L94-L143) builds a boolean mask called `content_word_mask` (`50,780` valid tokens) that keeps only space-prefixed alphabetic words while excluding 35 filler words defined in [`GRAMMAR_GLUE` (`pada/tree_engine/constants.py#L31-L37`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L31-L37):

```python
GRAMMAR_GLUE = {
    "a", "an", "the", "this", "that", "these", "those", "i",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "to", "of", "in", "for", "with", "on", "at", "from", "by",
    "into", "onto", "about", "through", "under", "over",
    "and", "or", "as", "than",
}
```

Masking out `GRAMMAR_GLUE` blocks the model from starting a sentence with `"The"` or `"It"` and forces the probability mass onto content nouns, verbs, and adjectives.

#### 2. Debugging the `'cello'` subword bug ([`is_bound_bpe_prefix` in `pada/tree_engine/lexicon.py#L69-L80`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/lexicon.py#L69-L80))
Most common English words (`"harvard"`, `"daniel"`, `"toronto"`, `"oxford"`) are single tokens in Qwen2.5, so their probabilities come straight out of Pass 1.

Multi-token words were trickier. Some words split into an incomplete stem plus a suffix, like `"locust"` (`['Ġloc', 'ust']`). When I first ran my reading comprehension test ([`scripts/test_story_comprehension.py`](https://github.com/Akintunde102/pada/blob/main/scripts/test_story_comprehension.py)) on a story about a man named Daniel who plays the cello, `Qwen2.5-1.5B` predicted **`"cell"`** instead of **`"cello"`**.

Checking the tokenizer output in Python showed why:
```python
>>> tokenizer.convert_ids_to_tokens(tokenizer.encode(" cello"))
['Ġc', 'ello']  # Token 272 (" c") + Token 4791 ("ello")
```
Qwen2.5 does not have a single token for `" cello"`. It splits it into a 1-letter space-prefixed piece `" c"` (`272`) followed by `"ello"` (`4791`). My original vocabulary filter in `lexicon.py` had `if len(clean) >= 2:` to block single letters from being returned as words. Because `" c"` had length 1, it was masked out, and the model fell back to the closest $\ge 2$-letter token: `" cell"` (`2779`).

To fix it without allowing single letters as final answers:
1. I changed the filter in [`pada/tree_engine/lexicon.py#L121-L125`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/lexicon.py#L121-L125) from `len(clean) >= 2` to `len(clean) >= 1`.
2. In [`is_bound_bpe_prefix()` (`pada/tree_engine/lexicon.py#L72-L73`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/lexicon.py#L72-L73), I marked every 1-letter space-prefixed token as a bound prefix (`bound_prefix_mask = True`, covering `538` bound stems total). In [`complete_all_open_fused()` (`pada/tree_engine/open_mode.py#L288-L294`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py#L288-L294), `forbid_bound_stop = True` prevents any bound prefix from stopping at step 0 and forces it to attach a subword continuation (`'Ġc' + 'ello'` $\rightarrow$ `"cello"`, scoring `94.7%`).
3. Finally, because opening `len(clean) >= 1` would also let the standalone 1-letter English words `"a"` and `"I"` through as word starters, and `"a"` was already in `GRAMMAR_GLUE`, I added `"i"` to [`GRAMMAR_GLUE` (`pada/tree_engine/constants.py#L32`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L32). That kept `"a"` and `"I"` blocked as conversational filler while letting 1-letter BPE prefixes like `" c"` start multi-token words.

#### 3. Where the engine is today vs. true single-pass multi-token extraction

Here is how the pipeline actually runs today in [`TreeEngine.predict()` (`pada/tree_engine/engine.py#L112-L295`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/engine.py#L112-L295):

```
CURRENT PIPELINE (Pass 1 Tree Prefill + Pass 2 Fused KV Continuation):
┌─────────────────────────────────────────────────────────────────────────┐
│ PASS 1 (pack_guided_tree in pada/tree_engine/tree_builder.py#L11):      │
│ Encodes Shared Story + All Q Branches (Open, Constrained, NOUL) at once │
│  ├─► First-token logits ("harvard", "toronto", "yes", "no") ready       │
│  └─► Saves tree_kv (batch_size = 1)                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ PASS 2 (complete_all_open_fused & score_all_constrained_fused):         │
│ Packs subword stems ('Ġc' -> 'ello') & P(stop) checks across ALL        │
│ questions directly onto tree_kv without re-encoding the story           │
└─────────────────────────────────────────────────────────────────────────┘
```

Single-token answers are resolved in Pass 1, while multi-token BPE words (`'Ġc' + 'ello'`) and $P(\text{stop})$ checks use Pass 2 ([`complete_all_open_fused` in `pada/tree_engine/open_mode.py#L193-L391`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py#L193-L391)) on top of Pass 1's cached `tree_kv`.

My longer-term goal for `Nameless` is to make even multi-token open extraction happen inside **a single forward pass (Pass 1)** without needing Pass 2 at all. Two existing techniques make that a realistic next step:
1. **Multi-Token Prediction (MTP) / Medusa subword heads:** Standard LLMs only project the final hidden state $h_{\text{readout}}$ through `lm_head` to predict the next token $t_1$. As shown in **Medusa** (Cai et al., 2024) and **DeepSeek-V3's Multi-Token Prediction** (Gloeckle et al., 2024; DeepSeek-AI, 2024), attaching 2 or 3 small parallel lookahead heads ($W^{(1)}, W^{(2)}, W^{(\text{stop})}$) to $h_{\text{readout}}$ lets one forward pass predict Subword 1 (`'Ġc'`), Subword 2 (`'ello'`), and $P(\text{stop})$ simultaneously from the same token position.
2. **Whole-word lexicon projection:** Even without training extra heads, you can pre-compose the `lm_head.weight` rows of known multi-token English words (`"cello"`, `"cardiology"`, `"locust"`) into a static word-level readout table so they score in a single matrix multiply at the end of Pass 1.

---

## 8. GPU profiling, speed benchmarks, and multi-domain test results

### Benchmark 1: What GPU profiling revealed ([`scripts/benchmark_speed_comparison.py`](https://github.com/Akintunde102/pada/blob/main/scripts/benchmark_speed_comparison.py))

When I first moved the codebase from CPU onto the local GPU in my laptop (an **NVIDIA Quadro P1000** with 4 GB VRAM, running in `float16`) and timed [`scripts/benchmark_speed_comparison.py`](https://github.com/Akintunde102/pada/blob/main/scripts/benchmark_speed_comparison.py), the numbers didn't add up at first:
* **Pass 1 (`Tree Prefill` for all 12 questions)** took **`589.6 ms`** (`49 ms/question`).
* **Pass 2a ([`score_all_constrained_fused`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L128) for all 7 Constrained and NOUL questions)** took **`354.3 ms`**.
* Yet full end-to-end `TreeEngine.predict()` took **`4,674.2 ms`**.

Profiling each step showed why. While I had already fused Constrained Mode into a single `batch_size = 1` pass on top of `tree_kv`, my initial Open Mode implementation ([`complete_batch_branched` in `pada/tree_engine/open_mode.py#L19-L164`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py#L19-L164)) was still being called inside a Python `for` loop once per Open question. For 5 Open questions, it was slicing the KV cache 5 separate times, duplicating it across `open_beam = 10` batch rows (`repeat_cache`), and running up to 5 decode steps per question.

Writing [`complete_all_open_fused()` (`pada/tree_engine/open_mode.py#L193-L391`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py#L193-L391) to pack all Open Mode subword continuations directly onto `tree_kv` (`batch_size = 1`, cropping back to $P$ after each step) cut full end-to-end GPU latency by **2.5×**, from **`4,674.2 ms` down to `1,892.7 ms`**.

#### Stage-by-stage GPU breakdown ($Q = 12$ mixed questions on `Quadro P1000`, `Qwen2.5-0.5B`)
| Pipeline Stage ([`TreeEngine.predict()`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/engine.py#L112)) | What Runs in This Stage | GPU Time (`ms`) | Share of Total |
| :--- | :--- | :--- | :--- |
| **Pass 1 ([`pack_guided_tree`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11))** | Encodes shared story + all 12 question branches in one 4D Tree Attention pass; produces first-token logits and `tree_kv` | **589.6 ms** (`49 ms/q`) | **29.4%** |
| **Pass 2a ([`score_all_constrained_fused`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L128))** | Scores candidate tokens + $P(\text{stop})$ across all 7 Constrained and NOUL questions in one pass on `tree_kv` | **354.3 ms** (`51 ms/q`) | **17.7%** |
| **Pass 2b ([`complete_all_open_fused`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py#L193))** | Completes multi-token BPE subwords (`'Ġc' + 'ello'`) + $P(\text{stop})$ across all 5 Open Mode questions on `tree_kv` | **1,062.3 ms** (`212 ms/q`) | **52.9%** |
| **Total** | **All 12 questions (5 Open + 3 Choice + 4 NOUL) + 2D OOD checks** | **2,006.2 ms** (`167 ms/q`) | **100.0%** |

---

#### End-to-end speed comparison: Packed Tree vs. Sequential Calls vs. JSON `.generate()` ([`scripts/benchmark_speed_comparison.py#L34-L210`](https://github.com/Akintunde102/pada/blob/main/scripts/benchmark_speed_comparison.py#L34-L210))
The benchmark script compares four measurements across $Q = 4, 8,$ and $12$ questions over the same story:
* **1a. Tree Prefill ([`pack_guided_tree`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L11)):** Pass 1 shared-trunk forward pass alone.
* **1b. Full [`TreeEngine.predict()`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/engine.py#L112):** Pass 1 + Pass 2a Fused Constrained + Pass 2b Fused Open Mode + $P(\text{stop})$ + OOD.
* **2. $Q$ Separate Sequential Calls:** Calling `model(story + question_i)` $Q$ times in a loop (re-encoding the story each time).
* **3. Autoregressive JSON `.generate()`:** Generating `{"key": "val"}` token-by-token (`6` tokens per question).

##### A. Local laptop GPU (`NVIDIA Quadro P1000`, 4 GB VRAM, `float16`)
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

On the 12-question story test with `Qwen2.5-1.5B`, moving from CPU (`39,652 ms`) to the `Quadro P1000` (`3,899 ms` on [`scripts/test_story_comprehension.py`](https://github.com/Akintunde102/pada/blob/main/scripts/test_story_comprehension.py), or `5,464 ms` inside the multi-run benchmark loop) gave an ~7×–10× speedup while keeping the same 11 confident passes and 1 borderline pass (`52.1%` in `fp32`, `51.9%` in `fp16`). On `0.5B` at $Q = 12$ on GPU, Pass 1 Tree Prefill (`581.2 ms`) is **21.4× faster** than JSON `.generate()` (`12,459.7 ms`), and the full `TreeEngine.predict()` (`1,892.7 ms`) is **6.58× faster**.

The `Quadro P1000` is a 4 GB mobile Pascal card (`sm_61`, 2017) without Tensor Cores. Because Tree Attention batches all $Q$ branches into dense matrix-matrix (`GEMM`) multiplies whereas `.generate()` still does 72 sequential memory-bandwidth reads, modern Tensor-Core GPUs (T4, L4, RTX 4090, A100) should widen that gap further, though I have only benchmarked my local CPU and `Quadro P1000` so far.

---

### Demo 2: 12-question reading comprehension walkthrough ([`scripts/test_story_comprehension.py`](https://github.com/Akintunde102/pada/blob/main/scripts/test_story_comprehension.py))
*(Methodology note: This 12-question story is a **development demo**, not a held-out benchmark. I used this story to catch the `'Ġc' + 'ello'` prefix bug and added the two boolean `yes`/`no` examples to [`GUIDED_ANCHOR_PROMPT` (`pada/tree_engine/constants.py#L57-L62`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constants.py#L57-L62) after seeing `0.5B` fail on boolean questions.)*

> **Story:** *"Amara Okafor graduated from Harvard with a degree in architecture before marrying Daniel Mensah in Lagos. Last autumn, the couple moved to Toronto so Amara could lead a museum renovation project. Even though Daniel enjoys playing the cello in his free time, he has been furious all week because the airline lost his luggage during the flight. Meanwhile, Amara's younger sister Chidinma still attends Oxford, where she studies robotics and volunteers at a local hospital on weekends."*

All 12 questions (5 Open, 3 Constrained, 4 NOUL) run together in one `TreeEngine.predict()` call:

| Question Key | Question Asked | Mode | Expected | `0.5B` Output (Score) | `1.5B` Output (Score) | `1.5B` Eval |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `amara_school_open` | Which university did Amara graduate from? | **OPEN** | `Harvard` | `harvard` (91.8%) | `harvard` (**95.7%**) | PASS |
| `spouse_first_name_open` | First name of the man Amara married? | **OPEN** | `Daniel` | `daniel` (94.5%) | `daniel` (**97.9%**) | PASS |
| `current_city_open` | Which city did the couple move to? | **OPEN** | `Toronto` | `toronto` (99.1%) | `toronto` (**99.3%**) | PASS |
| `sister_school_open` | Which university does her sister attend? | **OPEN** | `Oxford` | `oxford` (86.1%) | `oxford` (**98.0%**) | PASS |
| `instrument_open` | Which musical instrument does Daniel play? | **OPEN** | `cello` | `piano` (32.2%) | `cello` (**94.7%**) | PASS |
| `amara_profession_choice` | What field did Amara earn her degree in? | **CONSTRAINED** | `architecture` | `architecture` (99.7%) | `architecture` (**100.0%**) | PASS |
| `wedding_city_choice` | In which city did they get married? | **CONSTRAINED** | `Lagos` | `Toronto` (98.5%) | `Lagos` (**71.2%**) | PASS |
| `sister_major_choice` | What subject does Chidinma study? | **CONSTRAINED** | `robotics` | `robotics` (97.8%) | `robotics` (**99.9%**) | PASS |
| `noul_daniel_angry` | Is Daniel angry right now? | **NOUL (Y/N)** | `yes` | `no` (59.2%) | `yes` (**81.2%**) | PASS |
| `noul_amara_doctor` | Is Amara a medical doctor? | **NOUL (Y/N)** | `no` | `yes` (72.8%) | `no` (**97.4%**) | PASS |
| `noul_luggage_lost` | Did the airline lose Daniel's luggage? | **NOUL (Y/N)** | `yes` | `no` (88.5%) | `yes` (**73.4%**) | PASS |
| `noul_sister_in_toronto` | Does Chidinma attend school in Toronto? | **NOUL (Y/N)** | `no` | `yes` (71.7%) | `no` (**52.1%**) | Borderline *(near 50/50 coin flip)* |
| **Summary** | | | | **6 / 12** | **11 / 12 confident (+1 borderline)** | |

---

### Evaluation 3: The 11-scenario multi-domain suite ([`pada/tree_engine/suites.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/suites.py))
In [`pada/tree_engine/suites.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/suites.py), I tested both `Qwen2.5-0.5B` and `Qwen2.5-1.5B` across 11 scenarios: 7 Primary Dev scenarios (`23` accuracy-scored questions plus `2` unscored `expected=None` OOD negative-control questions in Scenario 7 = `25` questions) and 4 Secondary Domain scenarios (`12` questions across Hospitality, Port Logistics, University, and Agriculture), for **`35` accuracy-scored questions** (`37` total questions evaluated including the 2 OOD controls).

I broke the 35 scored questions down along two axes:
* **By mode:** `Open Mode` (`11` questions) vs. `Constrained Mode` (`24` questions).
* **By reasoning type:** `Extractive` (`13` questions where the answer word appears verbatim in the text, such as `"professor"` or `"maize"`) vs. `Inferential` (`22` questions where the answer word is not in the text and requires domain reasoning, such as `"cardiology"`, `"maritime"`, or `"indemnification"`). Constrained Mode scores lower than Open Mode on `0.5B` because 17 of the 24 Constrained questions are inferential traps (`wire` vs. `phishing`, `maritime` vs. `rail`, `sim` vs. `SIM`), whereas most Open Mode questions are extractive.

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

## 9. Running and testing the code yourself

Everything in [`github.com/Akintunde102/pada`](https://github.com/Akintunde102/pada) (Track 1, Track 2, speed benchmarks, training scripts, and all 33 unit tests) is under the [MIT License](https://github.com/Akintunde102/pada/blob/main/LICENSE):

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

### 1. Speed comparison and stage profiler ([`scripts/benchmark_speed_comparison.py`](https://github.com/Akintunde102/pada/blob/main/scripts/benchmark_speed_comparison.py))
```bash
python scripts/benchmark_speed_comparison.py --model-name Qwen/Qwen2.5-0.5B
```

### 2. 12-question story comprehension demo ([`scripts/test_story_comprehension.py`](https://github.com/Akintunde102/pada/blob/main/scripts/test_story_comprehension.py))
```bash
python scripts/test_story_comprehension.py --model-name Qwen/Qwen2.5-1.5B
```

### 3. 11-scenario suite and equivalence proofs ([`scripts/run_tree_engine.py`](https://github.com/Akintunde102/pada/blob/main/scripts/run_tree_engine.py))
```bash
# Run with float32 equivalence check (verifies max |tree_logits - seq_logits| < 1e-3):
python scripts/run_tree_engine.py --model-name Qwen/Qwen2.5-0.5B

# Or run on 1.5B (skipping verification to jump straight to the 11 scenarios):
python scripts/run_tree_engine.py --model-name Qwen/Qwen2.5-1.5B --skip-verify
```

### 4. Track 1 FastAPI server and training script ([`scripts/train_colab_1_5b.py`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py) and [`scripts/server.py`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py))
*(Large weight files in `checkpoints/` are gitignored, so running [`scripts/server.py`](https://github.com/Akintunde102/pada/blob/main/scripts/server.py) on a fresh clone falls back to base `Qwen2.5-1.5B` weights unless you run [`scripts/train_colab_1_5b.py`](https://github.com/Akintunde102/pada/blob/main/scripts/train_colab_1_5b.py) first to generate `checkpoints/jev_1_5b_lora_adapter` and `checkpoints/jev_1_5b_heads_and_T.pt`.)*
```bash
# Optional: train the Stage A + Stage B LoRA adapter & readout heads
python scripts/train_colab_1_5b.py

# Start the FastAPI server & Studio UI on http://127.0.0.1:8000
python scripts/server.py
```

### 5. Unit test suites (33 tests: 12 Schema in [`tests/test_schema.py`](https://github.com/Akintunde102/pada/blob/main/tests/test_schema.py) + 21 Tree Engine in [`tests/`](https://github.com/Akintunde102/pada/tree/main/tests))
```bash
python tests/test_schema.py
python -m unittest discover -s tests -p "test_tree_*.py"
```

---

## 10. Current limitations and what is left to build

1. **No ECE or reliability diagram calibration on Track 2 yet:** Track 1 trains its heads with Brier + Cross-Entropy loss and applies temperature scalars ($T$). Track 2 normalizes `lm_head` logits with $P(\text{stop})$. While $P(\text{stop})$ removes BPE length bias, I have not yet measured Expected Calibration Error (ECE) or fitted temperature scaling on a large labeled validation set for Track 2.
2. **OOD raw mass depends on candidate token length:** In Track 2, `is_ood` ([`pada/tree_engine/constrained.py#L38-L41`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/constrained.py#L38-L41)) uses a fixed raw probability mass threshold tuned on 3-to-4-candidate examples. Because joint probability naturally drops as candidate token length increases, production OOD detection needs thresholds conditioned on token length and option count.
3. **Single-word output boundary in Open Mode:** [`pada/tree_engine/open_mode.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/open_mode.py) is built for single-word answers (`"Harvard"`, `"cardiology"`, `"locust"`). Multi-word entities like `"New York City"` stop after the first complete word.
4. **Casing sensitivity in constrained candidates:** In Scenario 2 (SIM-swapping fraud in [`pada/tree_engine/suites.py`](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/suites.py)), the candidate list passed lowercase `"sim"`, while the model put 32.1% of its unconstrained probability mass on uppercase `"SIM"`. Marginalizing over lowercase, Title-case, and UPPERCASE BPE variants for each candidate is a straightforward fix I haven't added yet.
5. **Small dense models (`0.5B` / `1.5B`) on multi-hop negation:** On the indirect negation question (*"Chidinma attends Oxford... Does Chidinma attend school in Toronto?"*), `0.5B` got it wrong (`71.7% yes`) and `1.5B` barely passed (`52.1% no`). Larger models (`Qwen2.5-7B`) or open Sparse MoE models (`Qwen3-30B-A3B`, `DeepSeek-V3`) should handle multi-hop boolean hops better, though I have only tested `0.5B` and `1.5B` so far.
6. **Dense 4D attention mask allocation:** [`pack_guided_tree()` (`pada/tree_engine/tree_builder.py#L58-L59`)](https://github.com/Akintunde102/pada/blob/main/pada/tree_engine/tree_builder.py#L58-L59) currently builds a dense `[1, 1, N, N]` float tensor for PyTorch SDPA. Moving to PyTorch FlexAttention `BlockMask` or FlashInfer Cascade Attention would skip masked branch blocks at the CUDA kernel level.

---

## 11. References

* **TypeSafe AI's Jev and `archerhume`'s probe:**
  * [InfoQ: TypeSafe AI Releases Jev](https://infoq.com/news/2026/10/typesafe-ai-jev-released)
  * [The Register: TypeSafe AI Debuts Model for Machines](https://www.theregister.com/a/5296711)
  * [VKTR: ChatGPT Co-Creator Launches TypeSafe AI with Jev](https://www.vktr.com/ai-platforms/chatgpt-cocreator-launches-typesafe-ai-with-jev/)
  * [TechForge: Early Access and Funding Coverage](https://www.artificialintelligence-news.com/news/chatgpt-pioneer-launches-jev-model-for-programmatic-logic/)
  * [The New Stack: TypeSafe's Jev System One Model](https://thenewstack.io/typesafe-jev-system-one/)
  * [BestHub Republished Summary of `archerhume`'s 10,000-Call Jev Probe](https://www.besthub.dev/articles/reverse-engineering-jev-10k-api-calls-expose-closed-source-model-architecture-27085f944485)
  * [Sébastien Dubois: Reinforcement Learning for Calibrated Decisions (RLCD) & Rajesh Beri's Synthetic Ticket Evaluation](https://www.dsebastien.net/reinforcement-learning-for-calibrated-decisions-rlcd/)
  * [Towards AI: How Does Jev Work? A Technical Reconstruction from 5 Public Clues](https://pub.towardsai.net/how-does-jev-work-a-technical-reconstruction-from-5-public-clues-c32b29dcb3d5)
* **Open-source Jev reconstructions and comparisons:**
  * [Jared Palmer: `jaredpalmer/kev` (Open-Source Decision Model)](https://github.com/jaredpalmer/kev) & [The Menon Lab Analysis of Kev](https://themenonlab.blog/blog/kev-open-source-decision-model)
  * [TheoLeeCJ: `TheoLeeCJ/SemIf-OpenJev` (Semantic Ifs / OpenJev)](https://github.com/TheoLeeCJ/SemIf-OpenJev)
  * [Sagar Tamang: `SAGAR-TAMANG/sarvam-jev` Wiki](https://github.com/SAGAR-TAMANG/sarvam-jev/wiki)
  * [Kye Gomez: `kyegomez/open-jev`](https://github.com/kyegomez/open-jev)
  * [Lilting: Jev Clones Architecture Comparison (Kev, SemIf, Laya, Nimble, jevos)](https://lilting.ch/en/articles/jev-clones-architecture-comparison)
  * [Analytics Vidhya: Jev Projects on GitHub](https://www.analyticsvidhya.com/blog/2026/10/jev-projects-on-github/)
* **Verbalizers and log-likelihood scoring:**
  * Schick & Schütze (2021), *"Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference"* (EACL 2021).
  * Gao et al., *"A Framework for Few-Shot Language Model Evaluation"* (`lm-evaluation-harness`).
* **Calibration and latent knowledge in LLMs:**
  * Kadavath et al. (Anthropic), *"Language Models (Mostly) Know What They Know"* (arXiv:2207.05221).
  * Burns et al., *"Discovering Latent Knowledge in Language Models Without Supervision"* (ICLR 2023).
  * Glenn W. Brier (1950), *"Verification of forecasts expressed in terms of probability"*.
* **Tree Attention, shared-prefix inference, and multi-token prediction:**
  * Juravsky et al. (2024), *"Hydragen: High-Throughput LLM Inference with Shared Prefixes"*.
  * Yao et al. (2024), *"DeFT: Decoding with Flash Tree-Attention for Efficient Tree-Structured LLM Inference"*.
  * Zheng et al. (2024), *"SGLang: Efficient Execution of Structured Language Model Programs"* (RadixAttention).
  * Cai et al. (2024), *"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"*.
  * Gloeckle et al. (2024), *"Better & Faster Large Language Models via Multi-token Prediction"* (ICML 2024) & DeepSeek-AI (2024), *"DeepSeek-V3 Technical Report"* (Multi-Token Prediction / MTP).
  * PyTorch Blog, *"FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention"*.
