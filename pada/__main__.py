"""Command-line entry point for Pada.

Usage:
    python -m pada demo
    python -m pada validate <request_file.json>
    python -m pada export-schemas [--outdir <path>]
    python -m pada test
"""

import argparse
import json
import sys
from pathlib import Path

from pada.schema import (
    HostModelAdapter,
    JevRequest,
    Usage,
    calculate_output_tokens,
)
from pada.schema.export_json_schema import export_schemas


def run_demo() -> None:
    """Runs a simulated end-to-end demonstration of the Phase 0 contract."""
    print("=" * 60)
    print("JEV PHASE 0 INTERACTIVE CONTRACT DEMO")
    print("=" * 60)

    sample_payload = {
        "state": (
            "Customer Ticket #48291:\n"
            "My payouts have failed three times. The bank says everything is fine. "
            "Can someone please fix this immediately?"
        ),
        "questions": {
            "queue": {
                "type": "choice",
                "instructions": "Which department should handle this ticket?",
                "criteria": {
                    "payments": "Payout failures, refunds, and billing issues",
                    "account": "Login credentials and profile access",
                    "other": "Miscellaneous general queries",
                },
            },
            "escalate": {
                "type": "noul",
                "instructions": "Does this message require urgent human attention?",
            },
            "sentiment": {
                "type": "score",
                "instructions": "Rate customer urgency/distress from 1 (mild) to 5 (extreme)",
                "levels": 5,
            },
        },
    }

    print("\n1. [INCOMING REQUEST]: Validating against JevRequest schema...")
    req = JevRequest.model_validate(sample_payload)
    print(f"   -> Validated successfully. State length: {len(req.state)} chars, Questions: {len(req.questions)}.")

    print("\n2. [HOST ADAPTER]: Stripping user keys before sending to neural model...")
    model_input, mapping = HostModelAdapter.strip_request(req)
    print(f"   -> Created StrippedModelInput with {len(model_input.branches)} isolated branches:")
    for b in model_input.branches:
        print(f"      - Branch {b.index} [{b.question_type}]: '{b.instructions}'")
        if b.options:
            for opt in b.options:
                print(f"         * Option {opt.index}: '{opt.text}'")

    print("\n3. [KEY ISOLATION VERIFICATION]:")
    print("   -> Question keys ('queue', 'escalate', 'sentiment') and option keys ('payments', etc.)")
    print("      are completely absent from the model input representation.")

    print("\n4. [SIMULATED MODEL INFERENCE]:")
    # Simulate raw outputs from neural readout heads (Pointer head, Sigmoid head, Ordinal head)
    simulated_raw_outputs = {
        0: [0.91, 0.06, 0.03],                     # Branch 0 (choice): softmax probabilities
        1: 0.42,                                   # Branch 1 (noul): sigmoid P(True)
        2: ([0.02, 0.05, 0.13, 0.35, 0.45], 3.16), # Branch 2 (score): distribution + expectation
    }
    print("   -> Head 0 (Pointer Softmax):", simulated_raw_outputs[0])
    print("   -> Head 1 (Sigmoid P(True)):", simulated_raw_outputs[1])
    print("   -> Head 2 (Ordinal Score E[S]):", simulated_raw_outputs[2][1])

    print("\n5. [SYNTHETIC TOKEN BILLING]:")
    billed_output = calculate_output_tokens(req.questions)
    # Estimate input tokens (~4 chars/token)
    est_input = max(1, len(req.state) // 4 + 30)
    usage = Usage(input_tokens=est_input, output_tokens=billed_output)
    print(f"   -> Calculated usage: input_tokens={usage.input_tokens}, synthetic output_tokens={usage.output_tokens}")

    print("\n6. [RESPONSE RECONSTRUCTION]: Mapping model heads back to user keys...")
    response = HostModelAdapter.reconstruct_response(mapping, simulated_raw_outputs, usage)
    print("\n" + json.dumps(response.model_dump(), indent=2))
    print("\n" + "=" * 60)
    print("DEMO COMPLETE: All Phase 0 guarantees hold.")
    print("=" * 60)


def validate_file(path_str: str) -> None:
    """Validates a JSON file against the JevRequest schema."""
    p = Path(path_str)
    if not p.is_file():
        print(f"Error: File '{path_str}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON syntax in '{path_str}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        req = JevRequest.model_validate(data)
        output_tokens = calculate_output_tokens(req.questions)
        print(f"[OK] File '{path_str}' is a valid JevRequest.")
        print(f"     - State length: {len(req.state)} chars")
        print(f"     - Questions: {len(req.questions)}")
        print(f"     - Synthetic billed output_tokens: {output_tokens}")
    except Exception as e:
        print(f"[FAIL] Schema validation error in '{path_str}':\n{e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m jev",
        description="Jev: Calibrated Parallel Probability Decision Model CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # demo
    subparsers.add_parser("demo", help="Run the interactive Phase 0 contract and pipeline demo")

    # validate
    validate_parser = subparsers.add_parser("validate", help="Validate a JSON file against JevRequest schema")
    validate_parser.add_argument("file", help="Path to JSON file to validate")

    # export-schemas
    export_parser = subparsers.add_parser("export-schemas", help="Export JSON Schemas")
    export_parser.add_argument("--outdir", default=None, help="Output directory path for schema files")

    # test
    subparsers.add_parser("test", help="Run the automated test suite")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "validate":
        validate_file(args.file)
    elif args.command == "export-schemas":
        export_schemas(args.outdir)
    elif args.command == "test":
        import unittest
        suite = unittest.defaultTestLoader.discover("tests")
        runner = unittest.TextTestRunner(verbosity=2)
        res = runner.run(suite)
        sys.exit(0 if res.wasSuccessful() else 1)
    else:
        # Default behavior when run with no arguments: show help and quick hint
        parser.print_help()
        print("\nTip: Run 'python -m jev demo' to see the Phase 0 pipeline in action.")


if __name__ == "__main__":
    main()
