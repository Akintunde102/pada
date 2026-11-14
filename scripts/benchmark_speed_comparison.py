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
from pada.tree_engine.engine import TreeEngine
from scripts.test_story_comprehension import STORY, QUESTIONS


def _sync(device: str, clear_cache: bool = False):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        if clear_cache:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def run_benchmark(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    device: str = "auto",
    dtype_str: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype_str == "auto":
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
    else:
        dtype = getattr(torch, dtype_str)

    gpu_label = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "CPU"
    print(f"[*] Loading {model_name} on {device} ({gpu_label}, dtype={dtype})...")
    engine = TreeEngine.from_pretrained(model_name, device=device, dtype=dtype)
    tokenizer = engine.tokenizer
    model = engine.model
    masks = engine.masks

    # Warmup pass (especially important for CUDA kernel initialization)
    q_items = list(QUESTIONS.items())
    _ = engine.predict(state=STORY, questions=dict(q_items[:2]))
    _sync(device)

    print("\n" + "=" * 118)
    print(
        f"{'Q Count':<8} | {'1a. Tree Prefill':<18} | {'1b. Full TreeEngine':<20} | "
        f"{'2. Q Sequential Calls':<22} | {'3. JSON .generate()':<20} | {'Full Engine Speedup'}"
    )
    print("-" * 118)

    for q_count in (4, 8, 12):
        sub_q = dict(q_items[:q_count])

        # 1a. Packed Tree Pass (1 shared-trunk forward pass)
        _sync(device, clear_cache=True)
        t0 = time.perf_counter()
        ids_t, mask_4d, pos_t, _, S, _ = pack_guided_tree(
            state=STORY,
            questions=sub_q,
            tokenizer=tokenizer,
            id_to_word=masks.id_to_word,
            device=device,
            dtype=dtype,
        )
        with torch.inference_mode():
            out_1a = model(input_ids=ids_t, attention_mask=mask_4d, position_ids=pos_t, use_cache=False)
        _sync(device)
        tree_ms = (time.perf_counter() - t0) * 1000.0
        del out_1a, ids_t, mask_4d, pos_t

        # 1b. Full End-to-End TreeEngine.predict() (Pass 1 + Fused KV Continuation + P(stop) + OOD)
        _sync(device, clear_cache=True)
        t0 = time.perf_counter()
        _ = engine.predict(state=STORY, questions=sub_q)
        _sync(device)
        full_engine_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Q Separate Sequential Prefill Calls (re-encoding trunk + 1 question Q times)
        _sync(device, clear_cache=True)
        t0 = time.perf_counter()
        for k, v in sub_q.items():
            single_ids, _, single_pos, _, _, _ = pack_guided_tree(
                state=STORY,
                questions={k: v},
                tokenizer=tokenizer,
                id_to_word=masks.id_to_word,
                device=device,
                dtype=dtype,
            )
            with torch.inference_mode():
                out_seq = model(input_ids=single_ids, position_ids=single_pos, use_cache=False)
            del out_seq, single_ids, single_pos
        _sync(device)
        seq_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Autoregressive JSON .generate() (6 tokens per question to spell {"key": "val"})
        prompt = f"Story: {STORY}\nAnswer in JSON for {q_count} questions:\n{{"
        inp = {k: v.to(device) for k, v in tokenizer(prompt, return_tensors="pt").items()}
        n_gen_tokens = q_count * 6
        _sync(device, clear_cache=True)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out_gen = model.generate(
                **inp,
                max_new_tokens=n_gen_tokens,
                min_new_tokens=n_gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        _sync(device)
        gen_ms = (time.perf_counter() - t0) * 1000.0
        del out_gen, inp

        speedup_vs_seq = seq_ms / full_engine_ms
        speedup_vs_gen = gen_ms / full_engine_ms
        print(
            f"Q = {q_count:<4} | {tree_ms:>7.1f} ms ({tree_ms/q_count:>4.0f}/q) | "
            f"{full_engine_ms:>8.1f} ms ({full_engine_ms/q_count:>4.0f}/q) | "
            f"{seq_ms:>9.1f} ms ({seq_ms/q_count:>4.0f}/q) | "
            f"{gen_ms:>8.1f} ms ({n_gen_tokens:>2} tok) | "
            f"{speedup_vs_seq:.2f}x vs seq ({speedup_vs_gen:.2f}x vs JSON)"
        )
    print("=" * 118)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--dtype", default="auto", help="auto, float16, bfloat16, or float32")
    args = parser.parse_args()
    run_benchmark(args.model_name, device=args.device, dtype_str=args.dtype)
