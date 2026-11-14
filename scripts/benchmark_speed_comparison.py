"""Speed benchmark: Packed Tree Prefill vs. Q Sequential Prefills vs. Autoregressive JSON Decode.

Measures wall-clock latency across Q=4, 8, and 12 questions on the same shared story.
"""

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pada.tree_engine.tree_builder import pack_guided_tree
from pada.tree_engine.lexicon import build_vocab_masks
from scripts.test_story_comprehension import STORY, QUESTIONS


def run_benchmark(model_name: str = "Qwen/Qwen2.5-0.5B"):
    print(f"[*] Loading {model_name} for speed comparison...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    masks = build_vocab_masks(tokenizer, len(tokenizer))

    q_items = list(QUESTIONS.items())

    print("\n" + "=" * 95)
    print(f"{'Q Count':<8} | {'1. Packed Tree Pass':<22} | {'2. Q Sequential Calls':<24} | {'3. JSON .generate()':<22} | {'Tree Speedup'}")
    print("-" * 95)

    for q_count in (4, 8, 12):
        sub_q = dict(q_items[:q_count])

        # 1. Packed Tree Pass (1 shared-trunk forward pass)
        t0 = time.perf_counter()
        ids_t, mask_4d, pos_t, _, S, _ = pack_guided_tree(
            state=STORY,
            questions=sub_q,
            tokenizer=tokenizer,
            id_to_word=masks.id_to_word,
        )
        with torch.inference_mode():
            _ = model(input_ids=ids_t, attention_mask=mask_4d, position_ids=pos_t, use_cache=False)
        tree_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Q Separate Sequential Prefill Calls (re-encoding trunk + 1 question Q times)
        t0 = time.perf_counter()
        for k, v in sub_q.items():
            single_ids, _, single_pos, _, _, _ = pack_guided_tree(
                state=STORY,
                questions={k: v},
                tokenizer=tokenizer,
                id_to_word=masks.id_to_word,
            )
            with torch.inference_mode():
                _ = model(input_ids=single_ids, position_ids=single_pos, use_cache=False)
        seq_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Autoregressive JSON .generate() (5 tokens per question to spell {"key": "val"})
        prompt = f"Story: {STORY}\nAnswer in JSON for {q_count} questions:\n{{"
        inp = tokenizer(prompt, return_tensors="pt")
        n_gen_tokens = q_count * 6
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(
                **inp,
                max_new_tokens=n_gen_tokens,
                min_new_tokens=n_gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ms = (time.perf_counter() - t0) * 1000.0

        speedup_vs_seq = seq_ms / tree_ms
        speedup_vs_gen = gen_ms / tree_ms
        print(
            f"Q = {q_count:<4} | {tree_ms:>8.1f} ms ({tree_ms/q_count:>5.1f}/q) | "
            f"{seq_ms:>9.1f} ms ({seq_ms/q_count:>5.1f}/q) | "
            f"{gen_ms:>8.1f} ms ({n_gen_tokens:>2} tok) | "
            f"{speedup_vs_seq:.2f}x vs seq ({speedup_vs_gen:.2f}x vs JSON)"
        )
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    args = parser.parse_args()
    run_benchmark(args.model_name)
