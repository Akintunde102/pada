"""Evaluation harness, formatting, and result aggregation."""

from typing import List, Dict, Any, Tuple, Optional
from .engine import get_engine, TreeEngine


def display_evaluation(scenario_num: int, title: str, situation: str, out: Dict[str, Any]):
    """Pretty-print evaluation output table for a scenario."""
    results, metrics = out["results"], out["metrics"]
    print("\n" + "━" * 118)
    print(f"  SCENARIO {scenario_num}: {title.upper()}")
    print("━" * 118)
    print(f"Situation: \"{situation.strip()}\"\n")
    print(f"{'Question':<16} | {'Mode':<11} | {'Prediction':<15} | {'Conf':<7} | {'Mass / PremiseLL':<18} | {'Expected':<12} | {'Eval':<5} | Details")
    print("─" * 118)
    for q_key, ans in results.items():
        expected = f"'{ans['expected']}'" if ans["expected"] else "—"
        eval_str = ("PASS" if ans["correct"] else "FAIL") if ans["expected"] else "N/A"
        mass_str = f"{ans['raw_mass'] * 100:.3g}%"
        if ans.get("premise_ll") is not None:
            mass_str += f" ({ans['premise_ll']:.1f}P)"
        if ans["is_ood"]:
            mass_str += " OOD"
        if ans["type"] == "CONSTRAINED":
            items = sorted(ans["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
            details = "All: [" + ", ".join(f"{k}:{v * 100:3.0f}%" for k, v in items) + "]"
        else:
            details = "Tokens: [" + "+".join(ans["pieces"]) + f"]  content_mass={ans['content_mass'] * 100:.1f}%"
        print(
            f"{q_key[:15]:<16} | {ans['type']:<11} | {repr(ans['word']):<15} | {ans['confidence'] * 100:5.1f}% | "
            f"{mass_str:<18} | {expected:<12} | {eval_str:<5} | {details}"
        )
    print("─" * 118)
    for q_key, ans in results.items():
        if ans.get("diag"):
            print(ans["diag"])
    print(
        f"Performance: {metrics['num_questions']} questions in {metrics['latency_ms']} ms | "
        f"Trunk: {metrics['trunk_tokens']} tok | Total Tree: {metrics['total_packed_tokens']} tok | "
        f"1 tree pass + 1 fused candidate pass + pruned open completion"
    )


def evaluate_scenarios(
    scenarios: List[Dict[str, Any]],
    label: str,
    start_num: int = 1,
    engine: Optional[TreeEngine] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Evaluate a list of scenarios and collect evaluation records."""
    eng = engine or get_engine()
    records, total_ms = [], 0.0
    for idx, sc in enumerate(scenarios, start=start_num):
        out = eng.predict(sc["situation"], sc["questions"])
        display_evaluation(idx, sc["title"], sc["situation"], out)
        total_ms += out["metrics"]["latency_ms"]
        for q_key, ans in out["results"].items():
            spec = sc["questions"][q_key]
            rec = {
                "set": label,
                "key": q_key,
                "kind": spec.get("kind", "?"),
                "mode": ans["type"],
                "correct": ans["correct"] if ans["expected"] else None,
                "ood_ok": None,
                "xfail": bool(spec.get("xfail_ood")),
                "raw_mass": ans["raw_mass"],
                "premise_ll": ans.get("premise_ll"),
            }
            if "expect_ood" in spec:
                rec["ood_ok"] = ans["is_ood"] == spec["expect_ood"]
                verdict = (
                    ("PASS" if rec["ood_ok"] else "FAIL")
                    if not rec["xfail"]
                    else ("XPASS" if rec["ood_ok"] else "XFAIL")
                )
                prem_s = f", premise_ll={ans['premise_ll']:.2f}" if ans.get("premise_ll") is not None else ""
                print(
                    f"  OOD assertion [{q_key}]: expected_ood={spec['expect_ood']} got={ans['is_ood']} "
                    f"(raw_mass={ans['raw_mass']:.3g}{prem_s}, mass_thr={eng.ood_mass_threshold:.3g}, "
                    f"prem_thr={eng.ood_premise_threshold:.2f}) -> {verdict}"
                )
            records.append(rec)
    return records, total_ms


def summarize(records: List[Dict[str, Any]], total_ms: float, engine: Optional[TreeEngine] = None) -> bool:
    """Print overall summary table and assert OOD checks passed."""
    eng = engine or get_engine()

    def rate(rs):
        rs = [r for r in rs if r["correct"] is not None]
        return (sum(r["correct"] for r in rs), len(rs))

    print("\n" + "=" * 118)
    print("FINAL HARNESS EVALUATION SUMMARY")
    print("=" * 118)
    for label in ("dev", "heldout"):
        c, t = rate([r for r in records if r["set"] == label])
        if t:
            print(f"  • {label:<8} accuracy: {c}/{t} ({100 * c / t:.1f}%)")
    for kind in ("extractive", "inferential"):
        c, t = rate([r for r in records if r["kind"] == kind])
        if t:
            print(f"  • {kind:<11} accuracy: {c}/{t} ({100 * c / t:.1f}%)")
    for mode in ("CONSTRAINED", "OPEN"):
        c, t = rate([r for r in records if r["mode"] == mode])
        if t:
            print(f"  • {mode:<11} accuracy: {c}/{t} ({100 * c / t:.1f}%)")
    n_q = len(records)
    print(f"  • Total engine time: {total_ms:.1f} ms over {n_q} questions ({total_ms / max(n_q, 1):.1f} ms/q)")

    ood = [r for r in records if r["ood_ok"] is not None and not r["xfail"]]
    bad = [r for r in ood if not r["ood_ok"]]
    print(
        f"  • OOD assertions (Gross Mass + Hard Premise LL): {len(ood) - len(bad)}/{len(ood)} passed "
        f"(mass_thr={eng.ood_mass_threshold:.3g}, premise_thr={eng.ood_premise_threshold:.2f})"
    )
    correct_masses = [r["raw_mass"] for r in records if r["correct"] and r["mode"] == "CONSTRAINED"]
    correct_prems = [
        r["premise_ll"]
        for r in records
        if r["correct"] and r["mode"] == "CONSTRAINED" and r.get("premise_ll") is not None
    ]
    if correct_masses:
        print(f"  • Lowest raw mass among CORRECT constrained answers: {min(correct_masses):.3g}")
    if correct_prems:
        print(f"  • Lowest premise LL among CORRECT constrained answers: {min(correct_prems):.2f}")
    print("=" * 118 + "\n")
    return len(bad) == 0
