"""Main execution entrypoint for the Unified Tree Engine & Verification Suite.

Recalls all abstracted components from `pada.tree_engine`:
  - Initialization & Model Precision
  - Attention & Cache Equivalence Verification
  - 2D Zero-Pass OOD Calibration
  - Dev & Held-Out Multi-Scenario Benchmarking
  - Diagnostic inspection
"""

import argparse
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from pada.tree_engine import (
    init_engine,
    verify_tree_equivalence,
    verify_stage2_equivalence,
    calibrate_ood_threshold,
    evaluate_scenarios,
    summarize,
    diagnose,
    SCENARIOS,
    HELDOUT,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Tree Engine & Verification Runner")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="HuggingFace model ID or local directory path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device: 'cuda', 'cpu', or None for auto-detect",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip mathematical tree and stage-2 equivalence proofs",
    )
    parser.add_argument(
        "--skip-calibrate",
        action="store_true",
        help="Skip 2-D OOD threshold calibration",
    )
    parser.add_argument(
        "--suite",
        choices=["dev", "heldout", "both"],
        default="both",
        help="Which scenario suite to evaluate",
    )
    parser.add_argument(
        "--diag-key",
        type=str,
        default=None,
        help="Run diagnosis on a specific question key from scenario 1",
    )
    parser.add_argument(
        "--diag-mode",
        choices=["off", "fail", "all"],
        default="fail",
        help="Diagnostic trace output level",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 100)
    print("Unified Tree Engine (Stage 2 v4 Modular)")
    print(f"Loading Model: {args.model_name}")
    print("=" * 100)

    engine = init_engine(
        model_name=args.model_name,
        device=args.device,
        diag_mode=args.diag_mode,
    )

    if args.diag_key:
        print(f"\n[Diagnostic Mode] Diagnosing '{args.diag_key}' on Scenario 1...")
        diagnose(SCENARIOS[0]["situation"], SCENARIOS[0]["questions"], args.diag_key, engine=engine)
        return 0

    if not args.skip_verify:
        print("\n--- Running Mathematical Equivalence Proofs ---")
        for sc in (SCENARIOS[0], SCENARIOS[1]):
            verify_tree_equivalence(sc["situation"], sc["questions"], engine=engine)
        verify_stage2_equivalence(SCENARIOS[0]["situation"], SCENARIOS[0]["questions"], engine=engine)
        verify_stage2_equivalence(SCENARIOS[2]["situation"], SCENARIOS[2]["questions"], engine=engine)

    if not args.skip_calibrate:
        print("\n--- Running 2-D OOD Threshold Calibration ---")
        calibrate_ood_threshold(engine=engine, verbose=True)

    records, total_ms = [], 0.0

    if args.suite in ("dev", "both"):
        print(f"\n--- Evaluating Dev Suite ({len(SCENARIOS)} Scenarios) ---")
        dev_records, dev_ms = evaluate_scenarios(SCENARIOS, label="dev", start_num=1, engine=engine)
        records.extend(dev_records)
        total_ms += dev_ms

    if args.suite in ("heldout", "both"):
        print(f"\n--- Evaluating Held-out Suite ({len(HELDOUT)} Scenarios) ---")
        start_idx = len(SCENARIOS) + 1 if args.suite == "both" else 1
        held_records, held_ms = evaluate_scenarios(HELDOUT, label="heldout", start_num=start_idx, engine=engine)
        records.extend(held_records)
        total_ms += held_ms

    all_passed = summarize(records, total_ms, engine=engine)
    return 0 if all_passed else 1


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        main()
    else:
        sys.exit(main())
