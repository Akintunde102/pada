"""High-school literature style reading comprehension test for the Tree Engine.

Tests a rich narrative story with unambiguous factual questions across:
  1. Open Mode (no candidates — model must extract the exact word from the 151k vocab)
  2. Constrained Mode (multiple-choice candidates)
  3. NOUL Mode (binary Yes/No questions via candidates=["yes", "no"])
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pada.tree_engine import init_engine


STORY = (
    "Amara Okafor graduated from Harvard with a degree in architecture before marrying "
    "Daniel Mensah in Lagos. Last autumn, the couple moved to Toronto so Amara could lead "
    "a museum renovation project. Even though Daniel enjoys playing the cello in his free time, "
    "he has been furious all week because the airline lost his luggage during the flight. "
    "Meanwhile, Amara's younger sister Chidinma still attends Oxford, where she studies robotics "
    "and volunteers at a local hospital on weekends."
)

QUESTIONS = {
    # --- OPEN MODE (Zero candidates: must extract exact fact from full vocab) ---
    "amara_school_open": {
        "instructions": "Which university did Amara graduate from?",
        "expected": "Harvard",
    },
    "spouse_first_name_open": {
        "instructions": "What is the first name of the man Amara married?",
        "expected": "Daniel",
    },
    "current_city_open": {
        "instructions": "Which city did the couple move to last autumn?",
        "expected": "Toronto",
    },
    "sister_school_open": {
        "instructions": "Which university does Amara's younger sister attend?",
        "expected": "Oxford",
    },
    "instrument_open": {
        "instructions": "Which musical instrument does Daniel enjoy playing?",
        "expected": "cello",
    },

    # --- CONSTRAINED MODE (Multiple choice) ---
    "amara_profession_choice": {
        "instructions": "What field did Amara earn her degree in?",
        "candidates": ["architecture", "medicine", "law", "robotics"],
        "expected": "architecture",
    },
    "wedding_city_choice": {
        "instructions": "In which city did Amara and Daniel get married?",
        "candidates": ["Lagos", "Toronto", "London", "Accra"],
        "expected": "Lagos",
    },
    "sister_major_choice": {
        "instructions": "What subject does Chidinma study?",
        "candidates": ["robotics", "music", "architecture", "history"],
        "expected": "robotics",
    },

    # --- NOUL MODE (Yes/No binary questions via constrained ["yes", "no"]) ---
    "noul_daniel_angry": {
        "instructions": "Is Daniel angry right now?",
        "candidates": ["yes", "no"],
        "expected": "yes",
    },
    "noul_amara_doctor": {
        "instructions": "Is Amara a medical doctor?",
        "candidates": ["yes", "no"],
        "expected": "no",
    },
    "noul_luggage_lost": {
        "instructions": "Did the airline lose Daniel's luggage?",
        "candidates": ["yes", "no"],
        "expected": "yes",
    },
    "noul_sister_in_toronto": {
        "instructions": "Does Chidinma attend school in Toronto?",
        "candidates": ["yes", "no"],
        "expected": "no",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    args = parser.parse_args()

    engine = init_engine(model_name=args.model_name, diag_mode="fail")
    output = engine.predict(STORY, QUESTIONS)
    results = output["results"]

    print("\n" + "=" * 100)
    print("STORY:")
    print(f"  {STORY}")
    print("=" * 100)
    print(f"{'Question Key':<24} | {'Mode':<11} | {'Predicted':<15} | {'Conf':>6} | {'Expected':<14} | {'Eval'}")
    print("-" * 100)

    passed = 0
    total = len(QUESTIONS)
    for q_key, res in results.items():
        mode = "NOUL (Y/N)" if QUESTIONS[q_key].get("candidates") == ["yes", "no"] else res["type"]
        pred = res["word"]
        exp = res["expected"]
        ok = pred.lower() == exp.lower() or res["correct"]
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"{q_key:<24} | {mode:<11} | {pred:<15} | {res['confidence']*100:5.1f}% | {exp:<14} | {status}")

    print("-" * 100)
    print(f"SCORE: {passed}/{total} ({passed/total*100:.1f}%) | Latency: {output['metrics']['latency_ms']:.1f} ms")
    print("=" * 100)


if __name__ == "__main__":
    main()
