"""2-D OOD calibration (Raw Mass + Stage-1 Question Premise LL)."""

import math
from typing import Optional, List, Tuple
from .engine import get_engine, TreeEngine

CALIBRATION_CASES: List[Tuple[str, str, List[str], str]] = [
    # Positive controls (situation supports both the question premise and the candidates)
    (
        "The veterinarian splinted the dog's fractured leg after it was hit by a car.",
        "What body part was injured?",
        ["leg", "paw", "tail"],
        "pos",
    ),
    (
        "The meteorologist warned that a severe thunderstorm would reach the coast by evening.",
        "What weather event is expected?",
        ["thunderstorm", "drought", "blizzard"],
        "pos",
    ),
    (
        "The carpenter sanded the oak table and applied two coats of varnish.",
        "What material is the furniture made of?",
        ["wood", "plastic", "concrete"],
        "pos",
    ),
    # Gross negative controls (question premise is valid, but candidates are unrelated to both)
    (
        "The veterinarian splinted the dog's fractured leg after it was hit by a car.",
        "What body part was injured?",
        ["cryptocurrency", "parliament", "spreadsheet"],
        "gross_neg",
    ),
    (
        "The meteorologist warned that a severe thunderstorm would reach the coast by evening.",
        "What weather event is expected?",
        ["mortgage", "photosynthesis", "embezzlement"],
        "gross_neg",
    ),
    (
        "The carpenter sanded the oak table and applied two coats of varnish.",
        "What material is the furniture made of?",
        ["algorithm", "infantry", "vaccination"],
        "gross_neg",
    ),
    # Hard negative controls (candidates fit the question, but the question premise contradicts the situation)
    (
        "The veterinarian splinted the dog's fractured leg after it was hit by a car.",
        "What avian plumage was injured?",
        ["wing", "beak", "talon"],
        "hard_neg",
    ),
    (
        "The carpenter sanded the oak table and applied two coats of varnish.",
        "What culinary sauce was simmering?",
        ["bechamel", "marinara", "hollandaise"],
        "hard_neg",
    ),
]


def calibrate_ood_threshold(engine: Optional[TreeEngine] = None, verbose: bool = True) -> bool:
    """
    Calibrates BOTH thresholds at zero extra forward passes:
      1. ood_mass_threshold: geometric midpoint separating positive raw mass from gross negatives.
      2. ood_premise_threshold: midpoint separating positive question-premise log-likelihood from
         hard negatives (where the question's domain premise is ungrounded in the situation).
    """
    eng = engine or get_engine()
    pos_mass, gross_mass = [], []
    pos_prem, hard_prem = [], []

    for sit, q, cands, kind in CALIBRATION_CASES:
        out = eng.predict(sit, {"q": {"instructions": q, "candidates": cands}})
        r = out["results"]["q"]
        if kind == "pos":
            pos_mass.append(r["log_raw_mass"])
            pos_prem.append(r["premise_ll"])
        elif kind == "gross_neg":
            gross_mass.append(r["log_raw_mass"])
        elif kind == "hard_neg":
            hard_prem.append(r["premise_ll"])

    min_pos_m, max_gross_m = min(pos_mass), max(gross_mass)
    min_pos_p, max_hard_p = min(pos_prem), max(hard_prem)

    if verbose:
        print(f"[calibration] pos raw_mass:       {[f'{math.exp(x):.3g}' for x in pos_mass]}")
        print(f"[calibration] gross_neg raw_mass: {[f'{math.exp(x):.3g}' for x in gross_mass]}")
        print(f"[calibration] pos premise_ll:     {[f'{x:.2f}' for x in pos_prem]}")
        print(f"[calibration] hard_neg premise:   {[f'{x:.2f}' for x in hard_prem]}")

    ok = True
    if max_gross_m < min_pos_m:
        eng.ood_mass_threshold = math.exp((max_gross_m + min_pos_m) / 2)
    else:
        ok = False

    if max_hard_p < min_pos_p:
        eng.ood_premise_threshold = (max_hard_p + min_pos_p) / 2
    else:
        ok = False

    print(
        f"[calibration] 2D OOD thresholds -> MASS_THR = {eng.ood_mass_threshold:.3g}, "
        f"PREMISE_LL_THR = {eng.ood_premise_threshold:.2f}"
    )
    return ok
