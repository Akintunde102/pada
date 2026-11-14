# =====================================================================
# Nameless — 3-Mode 1-Pass Tree Engine (v6 — State-Only / Tags / Questions)
#
# Architecture:
#   1. Mode 1 (State-Only):
#      `predict(state)` -> outputs ONE single word via `Headline (one word):`
#      preceded by `STATE_ONLY_ANCHOR`.
#   2. Mode 2 (Tags):
#      `predict(state, tags=[...])` -> outputs 1 word per tag via `Tag [<name>]:`
#      preceded by `TAGS_ANCHOR` (with varied tag counts: 2, 5, 1, 3 tags per
#      example and zero overlap with test tags).
#   3. Mode 3 (Questions):
#      `predict(state, questions={...})` -> outputs 1 word per question via
#      `Question: <q> One-word answer:` preceded by `QA_ANCHOR` (with varied
#      question counts: 3, 1, 4, 2 questions per example).
#   4. True 1-Pass Execution:
#      - Constrained candidates (even multi-token) are packed directly as leaves
#        of the Stage-1 4D attention mask (`branch_seg` + `leaf_seg`).
#      - Open Mode uses Starter #1 (`k=1`, no stemming). Complete 1-token words
#        exit immediately after Pass #1 (0 extra passes); only bound BPE prefixes
#        ('Ġindemn', 'Ġcardi', 'Ġphotos', 'Ġarchae', 'Ġloc', 'Ġbot') run a
#        1-row continuation step to attach their suffix.
# =====================================================================
import contextlib
import math
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

try:
    from nltk.stem import SnowballStemmer
    _snowball = SnowballStemmer("english")

    def stem_word(w: str) -> str:
        return _snowball.stem(w.lower().strip())
except Exception:
    def stem_word(w: str) -> str:
        word = w.lower().strip()
        if len(word) > 5 and word.endswith("ing"):
            return word[:-3]
        if len(word) > 4 and word.endswith("ed"):
            return word[:-1] if word[:-1].endswith("e") else word[:-2]
        if len(word) > 4 and word.endswith("es"):
            return word[:-2]
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

KNOWN_BOUND_PREFIXES = {
    "emerg", "indemn", "card", "cardi", "cardiol", "orth", "orthop", "neurol", "neuro",
    "hosp", "hospit", "frust", "frustr", "crit", "damag", "repl", "replac",
    "invest", "investig", "malw", "ransom", "ransomw", "phish", "infring",
    "sever", "severab", "liab", "termin", "docum", "custom", "marit", "agric",
    "agricult", "infest", "univers", "prof", "profess", "housek", "bronch",
    "pulmon", "cardiov", "cardiovasc", "arrhythm", "infarct", "embol", "pneum",
    "pneumon", "pediat", "pediatr", "ergon", "ergonom", "obfusc", "unauth",
    "unauthor", "contam", "contamin", "degrad", "catast", "catastr", "immed",
    "loc", "photos", "bot", "archae", "power",
}

GRAMMAR_GLUE = {
    "a", "an", "the", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "to", "of", "in", "for", "with", "on", "at", "from", "by",
    "into", "onto", "about", "through", "under", "over",
    "and", "or", "as", "than", "you", "your", "my", "our", "his", "her", "their",
}

QUESTION_STOPWORDS = GRAMMAR_GLUE | {
    "what", "which", "who", "where", "when", "why", "how", "whose", "whom",
    "did", "does", "do", "has", "have", "had", "will", "would", "should",
    "could", "can", "may", "might", "must", "shall", "need", "needs",
    "one", "word", "answer", "question", "type", "kind", "level", "specific",
    "immediate", "primary", "main", "current", "right", "now", "there", "here",
    "handle", "handles", "handled", "take", "taken", "took", "make", "made",
    "cause", "caused", "affect", "affected", "receive", "received", "ask",
    "asked", "meet", "met", "target", "targeted", "cover", "covered", "tag",
}

active_model = None
active_tokenizer = None
device = "cpu"
dtype = torch.float32
vocab_size = 0
content_word_mask = None
bound_prefix_mask = None
subword_cont_mask = None
stop_boundary_mask = None
id_to_word = {}

P_STOP_FLOOR = 1e-4
OOD_MASS_THRESHOLD = 0.002
OOD_PREMISE_THRESHOLD = -6.5


def model_dtype():
    return next(active_model.parameters()).dtype


def _build_vocab_masks():
    global content_word_mask, bound_prefix_mask, subword_cont_mask, stop_boundary_mask, id_to_word, vocab_size
    vocab_size = max(active_model.config.vocab_size, len(active_tokenizer))
    content_word_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    bound_prefix_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    subword_cont_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    stop_boundary_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    id_to_word = {}

    tok_ids = list(range(len(active_tokenizer)))
    decoded = active_tokenizer.batch_decode([[i] for i in tok_ids], clean_up_tokenization_spaces=False)

    for tid, text in enumerate(decoded):
        if not text:
            continue
        if text[0] in (" ", "\n", ".", ",", ";", ":", "!", "?"):
            stop_boundary_mask[tid] = True
        if text.startswith(" "):
            raw = text[1:]
            clean = raw.lower()
            if raw.isalpha() and raw.isascii() and ((len(clean) >= 3 and clean not in GRAMMAR_GLUE) or clean in ("yes", "no")):
                content_word_mask[tid] = True
                id_to_word[tid] = clean
                if clean in KNOWN_BOUND_PREFIXES:
                    bound_prefix_mask[tid] = True
        else:
            if text.isalpha() and text.isascii() and (text.islower() or len(text) >= 2):
                subword_cont_mask[tid] = True
                id_to_word[tid] = text.lower()

    vocab = active_tokenizer.get_vocab()
    for eos_tok in ["<|im_end|>", "<|endoftext|>"]:
        if eos_tok in vocab and vocab[eos_tok] < vocab_size:
            stop_boundary_mask[vocab[eos_tok]] = True


def init_engine(model_name: str = "Qwen/Qwen2.5-1.5B", model=None, tokenizer=None):
    global active_model, active_tokenizer, device, dtype
    if model is None:
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
        print(f"Loading {model_name} on {device} ({dtype}, attn=sdpa)...")
        active_tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs = {"low_cpu_mem_usage": True, "attn_implementation": "sdpa"}
        try:
            m = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **load_kwargs)
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **load_kwargs)
        active_model = m.to(device)
    else:
        active_model = model
        active_tokenizer = tokenizer
        device = next(model.parameters()).device.type
        dtype = next(model.parameters()).dtype
    active_model.eval()
    _build_vocab_masks()


@contextlib.contextmanager
def precision(target: torch.dtype):
    orig = model_dtype()
    if orig != target:
        active_model.to(target)
    try:
        yield
    finally:
        if orig != target:
            active_model.to(orig)


def kv_pairs(cache):
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(l[0], l[1]) for l in cache]


def make_cache(pairs) -> DynamicCache:
    c = DynamicCache()
    for i, (k, v) in enumerate(pairs):
        c.update(k, v, i)
    return c


def slice_branch_kv(past_kv, trunk_len: int, branch_start: int, branch_end: int) -> DynamicCache:
    idx = torch.cat([
        torch.arange(0, trunk_len, device=device),
        torch.arange(branch_start, branch_end + 1, device=device),
    ])
    return make_cache([(k[:, :, idx, :].clone(), v[:, :, idx, :].clone()) for k, v in kv_pairs(past_kv)])


# =====================================================================
# 3 Format-Consistent Few-Shot Anchors (with Varied Counts per Example!)
# =====================================================================

# MODE 1 ANCHOR: State-Only -> 1 single word per situation
STATE_ONLY_ANCHOR = (
    "Task: Summarize each situation in a single lowercase word.\n\n"
    "Situation: The orbital observatory detected an intense gamma ray burst from a collapsing hypergiant.\n"
    "Headline (one word): discovery.\n\n"
    "Situation: The artisan pastry chef kneaded flour, chilled butter, and water to prepare a fruit tart.\n"
    "Headline (one word): pastry.\n\n"
    "Situation: The hiker ignored the storm warning and was rescued by helicopter after twelve hours on the mountain.\n"
    "Headline (one word): rescue.\n\n"
    "Situation: The accountant discovered that the quarterly report double counted two large invoices.\n"
    "Headline (one word): error.\n\n"
)

# MODE 2 ANCHOR: Tags -> Varied tag counts per example (2 tags, 5 tags, 1 tag, 3 tags) & diverse unseen tag names!
TAGS_ANCHOR = (
    "Task: For each tag, output a single lowercase word that fits the situation.\n\n"
    "Situation: Thank you so much for resolving my billing issue so quickly, you made my day!\n"
    "Tag [sentiment]: grateful.\n"
    "Tag [sarcasm]: no.\n\n"
    "Situation: This is ridiculous! Your courier threw my glass monitor over the fence and shattered it!\n"
    "Tag [mood]: furious.\n"
    "Tag [item]: monitor.\n"
    "Tag [severity]: high.\n"
    "Tag [refund]: yes.\n"
    "Tag [department]: shipping.\n\n"
    "Situation: Heavy snowfall closed the mountain highway and trapped twelve commuters overnight.\n"
    "Tag [weather]: blizzard.\n\n"
    "Situation: Pay the ransom in bitcoin by midnight or we will leak your company database.\n"
    "Tag [attitude]: hostile.\n"
    "Tag [threat]: yes.\n"
    "Tag [slur]: no.\n\n"
)

# MODE 3 ANCHOR: Questions -> Varied question counts per example (3 questions, 1 question, 4 questions, 2 questions)
QA_ANCHOR = (
    "Task: Answer each question about the situation in a single lowercase word.\n\n"
    "Situation: The orbital observatory detected an intense gamma ray burst from a collapsing hypergiant.\n"
    "Question: What astronomical instrument made the discovery? One-word answer: telescope.\n"
    "Question: What celestial object underwent collapse? One-word answer: star.\n"
    "Question: What electromagnetic emission was recorded? One-word answer: gamma.\n\n"
    "Situation: The marathon runner sprained his ankle at mile twenty and had to withdraw from the race.\n"
    "Question: Which body part was injured? One-word answer: ankle.\n\n"
    "Situation: The artisan pastry chef kneaded flour, chilled butter, and water to prepare a fruit tart in the bakery.\n"
    "Question: What baked pastry is being crafted? One-word answer: tart.\n"
    "Question: What dairy ingredient was incorporated? One-word answer: butter.\n"
    "Question: What culinary specialist prepared the dough? One-word answer: chef.\n"
    "Question: Where is the pastry being prepared? One-word answer: bakery.\n\n"
    "Situation: A sudden power outage halted the assembly line at the automotive factory for three hours.\n"
    "Question: What event stopped the machinery? One-word answer: outage.\n"
    "Question: What kind of facility was affected? One-word answer: factory.\n\n"
)


def _encode_candidate(c: str):
    return active_tokenizer.encode(" " + c.replace("_", " ").strip(), add_special_tokens=False)


def _find_question_premise_offsets(branch_ids: list) -> list:
    suffix_len = len(active_tokenizer.encode(" One-word answer:", add_special_tokens=False))
    prefix_len = len(active_tokenizer.encode("Question:", add_special_tokens=False))
    body_end = max(prefix_len + 1, len(branch_ids) - suffix_len)

    premise_offsets = []
    for offset in range(prefix_len, body_end):
        tid = branch_ids[offset]
        w = id_to_word.get(tid, active_tokenizer.decode([tid]).strip().lower())
        if w and w.isalpha() and len(w) >= 3 and w not in QUESTION_STOPWORDS:
            premise_offsets.append(offset)
    if not premise_offsets:
        premise_offsets = list(range(prefix_len, body_end))
    return premise_offsets


# =====================================================================
# 4. 3-Mode 1-Pass Tree Packer
# =====================================================================


def pack_three_mode_tree(state: str, questions=None, tags=None):
    """
    Selects the exact matching anchor and branch syntax for the 3 modes:
      - Mode 1 (State-Only): `questions is None and tags is None`
          Anchor: `STATE_ONLY_ANCHOR`, Single Branch: `Headline (one word):`
      - Mode 2 (Tags): `tags is not None`
          Anchor: `TAGS_ANCHOR`, Branches: `Tag [<tag_name>]:`
      - Mode 3 (Questions): `questions is not None`
          Anchor: `QA_ANCHOR`, Branches: `Question: <q> One-word answer:`
    Also packs any multi-token `candidates` directly as 1-pass leaf segments!
    """
    if tags is not None:
        mode = "TAGS"
        anchor = TAGS_ANCHOR
        if isinstance(tags, (list, tuple)):
            items = {t: {"tag": t} for t in tags}
        else:
            items = {k: (v if isinstance(v, dict) else {"tag": str(v)}) for k, v in tags.items()}
    elif questions is not None:
        mode = "QUESTIONS"
        anchor = QA_ANCHOR
        if isinstance(questions, (list, tuple)):
            items = {q: {"instructions": q} for q in questions}
        else:
            items = {k: (v if isinstance(v, dict) else {"instructions": str(v)}) for k, v in questions.items()}
    else:
        mode = "STATE_ONLY"
        anchor = STATE_ONLY_ANCHOR
        items = {"headline": {"raw_branch": "Headline (one word):"}}

    trunk_text = anchor + f"Situation: {state.strip()}\n"
    trunk_ids = active_tokenizer.encode(trunk_text, add_special_tokens=False)
    S = len(trunk_ids)

    all_ids = list(trunk_ids)
    branch_seg = [0] * S
    leaf_seg = [0] * S
    pos_ids = list(range(S))
    branch_metadata = {}

    next_cid = 1
    for branch_idx, (key, spec) in enumerate(items.items(), start=1):
        if mode == "STATE_ONLY":
            b_text = "Headline (one word):"
        elif mode == "TAGS":
            tag_label = spec.get("tag", spec.get("instructions", key)).strip().lower()
            b_text = f"Tag [{tag_label}]:"
        else:
            instr = spec.get("instructions", key).strip()
            b_text = f"Question: {instr} One-word answer:"

        branch_ids = active_tokenizer.encode(b_text, add_special_tokens=False)
        B_len = len(branch_ids)
        branch_start = len(all_ids)
        all_ids.extend(branch_ids)
        branch_end = len(all_ids) - 1

        branch_seg.extend([branch_idx] * B_len)
        leaf_seg.extend([0] * B_len)
        pos_ids.extend([S + i for i in range(B_len)])

        candidates = spec.get("candidates")
        cand_layout = []
        if candidates:
            base_pos = S + B_len
            for cand in candidates:
                toks = _encode_candidate(cand)
                c_start = len(all_ids)
                all_ids.extend(toks)
                branch_seg.extend([branch_idx] * len(toks))
                leaf_seg.extend([next_cid] * len(toks))
                pos_ids.extend([base_pos + j for j in range(len(toks))])
                cand_layout.append((cand, toks, c_start))
                next_cid += 1

        branch_metadata[key] = {
            "mode": mode,
            "branch_idx": branch_idx,
            "branch_start": branch_start,
            "branch_end": branch_end,
            "readout_idx": branch_end,
            "branch_len": B_len,
            "branch_ids": branch_ids,
            "premise_offsets": _find_question_premise_offsets(branch_ids) if mode == "QUESTIONS" else [],
            "candidates": candidates,
            "cand_layout": cand_layout,
            "expected": spec.get("expected"),
            "accept": spec.get("accept"),
            "spec": spec,
        }

    total_len = len(all_ids)
    b_seg = torch.tensor(branch_seg, dtype=torch.long, device=device)
    l_seg = torch.tensor(leaf_seg, dtype=torch.long, device=device)
    row = torch.arange(total_len, device=device).unsqueeze(1)
    col = torch.arange(total_len, device=device).unsqueeze(0)

    same_branch = (b_seg.unsqueeze(1) == b_seg.unsqueeze(0))
    valid_leaf = (l_seg.unsqueeze(0) == 0) | (l_seg.unsqueeze(1) == l_seg.unsqueeze(0))
    allowed = (col <= row) & ((b_seg.unsqueeze(0) == 0) | (same_branch & valid_leaf))

    mdt = model_dtype()
    mask = torch.zeros((1, 1, total_len, total_len), dtype=mdt, device=device)
    mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(mdt).min)

    ids_t = torch.tensor([all_ids], dtype=torch.long, device=device)
    pos_t = torch.tensor([pos_ids], dtype=torch.long, device=device)
    return ids_t, mask, pos_t, branch_metadata, S, mode


def compute_question_premise_ll(tree_logits: torch.Tensor, meta: dict) -> float:
    b_start = meta["branch_start"]
    b_ids = meta["branch_ids"]
    offsets = meta["premise_offsets"]
    lls = []
    for off in offsets:
        if off == 0:
            continue
        pred_pos = b_start + off - 1
        target_tid = b_ids[off]
        logp = F.log_softmax(tree_logits[0, pred_pos].float(), dim=-1)
        lls.append(float(logp[target_tid].item()))
    return sum(lls) / max(len(lls), 1) if lls else 0.0


def _finalize_constrained_dict(candidates: list, log_joints: dict, details: dict, premise_ll: float = None):
    mx = max(log_joints.values())
    log_raw_mass = mx + math.log(sum(math.exp(v - mx) for v in log_joints.values()))
    raw_mass = math.exp(log_raw_mass)
    prob_dict = {c_: round(math.exp(v - log_raw_mass), 4) for c_, v in log_joints.items()}
    sorted_c = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    winner = sorted_c[0][0]
    if premise_ll is not None:
        is_ood = (raw_mass < OOD_MASS_THRESHOLD) or (premise_ll < OOD_PREMISE_THRESHOLD)
    else:
        is_ood = raw_mass < OOD_MASS_THRESHOLD
    return {
        "word": winner, "confidence": sorted_c[0][1],
        "runner_up": sorted_c[1] if len(sorted_c) > 1 else None,
        "raw_mass": raw_mass, "log_raw_mass": log_raw_mass,
        "premise_ll": premise_ll, "is_ood": is_ood,
        "probabilities": prob_dict, "log_joints": log_joints, "details": details,
    }


def _read_single_pass_candidates(tree_logits: torch.Tensor, meta: dict):
    step1 = F.log_softmax(tree_logits[0, meta["readout_idx"]].float(), dim=-1)
    log_joints, details = {}, {}
    for cand, toks, c_start in meta["cand_layout"]:
        m = len(toks)
        lj = float(step1[toks[0]].item())
        for j in range(m - 1):
            token_logp = F.log_softmax(tree_logits[0, c_start + j].float(), dim=-1)
            lj += float(token_logp[toks[j + 1]].item())
        last_logp = F.log_softmax(tree_logits[0, c_start + m - 1].float(), dim=-1)
        p_stop = float(torch.exp(last_logp[stop_boundary_mask]).sum().item())
        lj += math.log(max(p_stop, P_STOP_FLOOR))
        log_joints[cand] = lj
        details[cand] = {
            "pieces": [active_tokenizer.convert_ids_to_tokens(int(t)) for t in toks],
            "p_first": math.exp(float(step1[toks[0]].item())),
            "p_stop": p_stop, "log_joint": lj,
        }
    return log_joints, details


def complete_starter_one(tid0: int, p_first: float, isolated_kv: DynamicCache, max_tokens: int = 5):
    P = isolated_kv.get_seq_length()
    cache = make_cache([(k.clone(), v.clone()) for k, v in kv_pairs(isolated_kv)])
    pieces = [id_to_word[tid0]]
    cur_tid = tid0
    comp_lj = 0.0
    for s in range(max_tokens):
        cur = torch.tensor([[cur_tid]], dtype=torch.long, device=device)
        pos = torch.full((1, 1), P + s, dtype=torch.long, device=device)
        with torch.inference_mode():
            out = active_model(input_ids=cur, past_key_values=cache, position_ids=pos, use_cache=True)
        probs = F.softmax(out.logits[0, -1, :].float(), dim=-1)
        ps = float(probs[stop_boundary_mask].sum().item())
        cont = probs.masked_fill(~subword_cont_mask, 0.0)
        cs = float(cont.sum().item())
        bp0, bt0 = cont.max(dim=0)
        bp0, bt0 = float(bp0.item()), int(bt0.item())
        is_bound_root = (s == 0 and bool(bound_prefix_mask[tid0].item()) and bp0 > 1e-6)
        if (not is_bound_root) and (ps >= cs or s == max_tokens - 1 or bp0 <= 0.0):
            comp_lj += math.log(max(ps, P_STOP_FLOOR))
            break
        piece_str = id_to_word.get(bt0, active_tokenizer.decode([bt0]).strip().lower())
        pieces.append(piece_str)
        comp_lj += math.log(max(bp0, 1e-12))
        cur_tid = bt0
    del cache
    return "".join(pieces), pieces, p_first * math.exp(comp_lj)


def _check_match(word: str, expected=None, accept=None) -> bool:
    if accept:
        return (word in accept) or any(stem_word(word) == stem_word(a) for a in accept)
    if expected:
        if isinstance(expected, (set, list, tuple)):
            return (word in expected) or any(stem_word(word) == stem_word(a) for a in expected)
        return stem_word(word) == stem_word(expected) or word == expected
    return False


def predict_unified_tree(state: str, questions=None, tags=None, top_k: int = 5):
    t0 = time.perf_counter()
    ids_t, mask_4d, pos_t, branch_meta, trunk_len, mode = pack_three_mode_tree(state, questions=questions, tags=tags)

    has_open_branch = any(not m["candidates"] for m in branch_meta.values())
    with torch.inference_mode():
        outputs = active_model(input_ids=ids_t, attention_mask=mask_4d, position_ids=pos_t, use_cache=has_open_branch)

    forward_passes = 1
    results = {}
    for q_key, meta in branch_meta.items():
        candidates = meta["candidates"]
        expected = meta["expected"]
        accept = meta["accept"]
        branch_logits = outputs.logits[0, meta["readout_idx"], :]

        if candidates:
            lj_dict, det_dict = _read_single_pass_candidates(outputs.logits, meta)
            premise_ll = compute_question_premise_ll(outputs.logits, meta) if meta["premise_offsets"] else None
            res = _finalize_constrained_dict(candidates, lj_dict, det_dict, premise_ll=premise_ll)
            is_match = _check_match(res["word"], expected=expected, accept=accept)
            results[q_key] = {
                "type": "CONSTRAINED", "word": res["word"], "confidence": res["confidence"],
                "raw_mass": res["raw_mass"], "log_raw_mass": res["log_raw_mass"],
                "premise_ll": res["premise_ll"], "is_ood": res["is_ood"],
                "runner_up": res["runner_up"], "expected": expected,
                "correct": is_match, "probabilities": res["probabilities"],
            }
        else:
            full_logp = F.log_softmax(branch_logits.float(), dim=-1)
            probs = torch.exp(full_logp)
            content_mass = float(probs[content_word_mask].sum().item())
            masked = branch_logits.float().masked_fill(~content_word_mask, float("-inf"))
            top_ids = torch.topk(masked, k=max(top_k, 2)).indices.tolist()
            tid0 = top_ids[0]
            p_first = float(probs[tid0].item())

            if not bool(bound_prefix_mask[tid0].item()):
                word = id_to_word[tid0]
                pieces = [word]
                raw_mass = p_first
            else:
                isolated_kv = slice_branch_kv(outputs.past_key_values, trunk_len, meta["branch_start"], meta["branch_end"])
                word, pieces, raw_mass = complete_starter_one(tid0, p_first, isolated_kv)
                forward_passes += 1
                del isolated_kv

            top_k_dict = {}
            for tid in top_ids[:top_k]:
                w_cand = id_to_word[tid] if tid != tid0 else word
                if w_cand not in top_k_dict:
                    top_k_dict[w_cand] = round(float(probs[tid].item()) / max(content_mass, 1e-12), 4)
            items = list(top_k_dict.items())
            conf = items[0][1] if items else round(p_first / max(content_mass, 1e-12), 4)
            runner_up = items[1] if len(items) > 1 else None
            is_match = _check_match(word, expected=expected, accept=accept)

            results[q_key] = {
                "type": mode, "word": word, "pieces": pieces,
                "confidence": conf, "raw_mass": raw_mass, "content_mass": round(content_mass, 4),
                "premise_ll": None, "is_ood": None,
                "runner_up": runner_up, "expected": expected, "correct": is_match, "top_k": top_k_dict,
            }

    del outputs
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "mode": mode,
        "results": results,
        "metrics": {
            "trunk_tokens": trunk_len, "total_packed_tokens": ids_t.shape[1],
            "num_branches": len(branch_meta), "forward_passes": forward_passes,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        },
    }


if __name__ == "__main__":
    init_engine()
    # Mode 1: state only (1 word)
    print(predict_unified_tree("I hate you"))
    # Mode 2: varied tag counts (4, 1, 6)
    s = "Thank you so much for staying late to fix the database server!"
    for tg in (["tone", "aggression", "f word", "n word"], ["urgency"],
               ["tone", "gratitude", "urgency", "profanity", "sarcasm", "domain"]):
        print(len(tg), predict_unified_tree(s, tags=tg)["results"].keys())
    # Mode 3: varied question counts (1, 2, 4)
    qs = ["What field is this about?", "Who was thanked?", "What was fixed?", "When did this happen?"]
    for n in (1, 2, 4):
        print(n, {k: v["word"] for k, v in predict_unified_tree(s, questions=qs[:n])["results"].items()})
