# =============================================================================
# Markov Logic Networks (MLN) — example 2
#
# LITERATURE
# ----------
# Original MLN paper:
#   Richardson, M., & Domingos, P. (2006).
#   Markov logic networks.
#   Machine Learning, 62(1–2), 107–136.
#   https://doi.org/10.1007/s10994-006-5833-1
#
# PARAMETER MAPPING
# -----------------
# Joint distribution over possible worlds  (Richardson & Domingos 2006, §2):
#   An MLN M = {(F_i, w_i)} defines a log-linear distribution:
#       P(X = x) = (1/Z) exp( Σ_i  w_i · n_i(x) )
#   where n_i(x) is the number of true groundings of formula F_i in world x,
#   and Z = Σ_x exp(Σ_i w_i n_i(x)) is the partition function.
#
# Rule weights — RULE_WEIGHT  (Richardson & Domingos 2006, §2 and §5):
#   Same setting as example 1: RULE_WEIGHT = 2.0.
#
# Unary log-odds potentials — soft evidence from membership values:
#   (Pearl 1988; Richardson & Domingos 2006, §6)
#
# Inference — exact enumeration  (Richardson & Domingos 2006, §4):
#   Example 2 uses 12 input + 4 output atoms → 2^16 = 65 536 worlds.
#   Exact enumeration remains tractable and is preferred for correctness.
#
# Score aggregation — expected value:
#       score = Σ_head  P(head) × EV(consequent(head))
# =============================================================================
from __future__ import annotations

import itertools
import math


RULE_WEIGHT: float = 2.0  # Richardson & Domingos 2006, §2 and §5

# ---------------------------------------------------------------------------
# Each rule encodes one IF-THEN clause from the original rule set.
# "antecedents" are the atom names for the AND-joined conditions in the rule
# body; "head" is the atom name for the THEN consequent.
# ---------------------------------------------------------------------------
RULES = [
    {
        "name": "r1",
        "text": "IF Italian IS Medium AND Portuguese IS Medium AND English IS Medium AND Vram IS Low THEN Weight IS High",
        "antecedents": ["it_medium", "pt_medium", "en_medium", "vram_low"],  # rule body
        "consequent": "High",
        "head": "weight_high",
        "weight": RULE_WEIGHT,
    },
    {
        "name": "r2",
        "text": "IF Italian IS Medium AND Portuguese IS Medium AND English IS Medium AND Oss IS High THEN Weight IS Very High",
        "antecedents": ["it_medium", "pt_medium", "en_medium", "oss_high"],  # rule body
        "consequent": "Very High",
        "head": "weight_very_high",
        "weight": RULE_WEIGHT,
    },
    {
        "name": "r3",
        "text": "IF Italian IS High AND Portuguese IS Medium AND English IS Medium THEN Weight IS High",
        "antecedents": ["it_high", "pt_medium", "en_medium"],  # rule body
        "consequent": "High",
        "head": "weight_high",
        "weight": RULE_WEIGHT,
    },
    {
        "name": "r4",
        "text": "IF Italian IS Low AND Portuguese IS Medium AND English IS Medium AND Vram IS High AND Oss IS Low THEN Weight IS Very Low",
        "antecedents": ["it_low", "pt_medium", "en_medium", "vram_high", "oss_low"],  # rule body
        "consequent": "Very Low",
        "head": "weight_very_low",
        "weight": RULE_WEIGHT,
    },
    {
        "name": "r5",
        "text": "IF English IS High AND Oss IS Medium AND Vram IS Medium THEN Weight IS Low",
        "antecedents": ["en_high", "oss_medium", "vram_medium"],  # rule body
        "consequent": "Low",
        "head": "weight_low",
        "weight": RULE_WEIGHT,
    },
]

# Defuzzification expected values EV = ∫tμ(t)dt / ∫μ(t)dt for each output label.
# Trapezoid params (a,b,c,d) used to compute these values:
#   Very Low  → (0.00, 0.00, 1.00, 2.25)  → EV = 0.852564
#   Low       → (1.00, 2.25, 3.25, 4.50)  → EV = 2.750000
#   Medium    → (3.25, 4.50, 5.50, 6.75)  → EV = 5.000000
#   High      → (5.50, 6.75, 7.75, 9.00)  → EV = 7.250000
#   Very High → (7.75, 9.00, 10.00, 10.00) → EV = 9.147436
# Source: compute_output_weight_integrals.py
EV_TABLE: dict[str, float] = {
    "Very Low":  0.852564,
    "Low":       2.750000,
    "Medium":    5.000000,
    "High":      7.250000,
    "Very High": 9.147436,
}

# Input membership values per model — these become the unary log-odds potentials
# (Pearl 1988; Richardson & Domingos 2006, §6).
# Each value: max-min overlap degree between the model's trapezoidal score
# distribution trap(mean±2σ, mean±0.5σ) and the linguistic MF for that label.
# Input MF params (a,b,c,d) used to compute these values:
#   EN Low  (2.823, 2.823, 7.565, 14.679)  EN Med (7.565, 14.679, 19.421, 26.535)  EN High (19.421, 26.535, 31.277, 31.277)
#   IT Low  (4.899, 4.899, 7.982, 12.608)  IT Med (7.982, 12.608, 15.692, 20.318)  IT High (15.692, 20.318, 23.401, 23.401)
#   PT Low  (1.200, 1.200, 5.792, 12.679)  PT Med (5.792, 12.679, 17.271, 24.158)  PT High (17.271, 24.158, 28.750, 28.750)
#   VRAM Low (8.0, 8.0, 16.0, 20.0)  VRAM Med (16.0, 20.0, 24.0, 28.0)  VRAM High (24.0, 28.0, 32.0, 32.0)
#   OSS Low  (0.0, 0.0, 3.0, 5.0)    OSS Med  (3.0, 5.0, 7.0, 9.0)      OSS High  (7.0, 9.0, 10.0, 10.0)
# Source: compute_trapezoidal_memberships.build_memberships()
MEMBERSHIPS: dict[str, dict[str, float]] = {
    "MichelRosselli/apertus": {
        "IT_High":    0.114389,
        "IT_Medium":  0.913492,
        "IT_Low":     0.953937,
        "PT_Medium":  0.713399,
        "EN_High":    0.000000,
        "EN_Medium":  0.753163,
        "VRAM_Low":   1.000000,
        "VRAM_Medium": 0.000000,
        "VRAM_High":  0.000000,
        "OSS_High":   1.000000,
        "OSS_Medium": 0.000000,
        "OSS_Low":    0.000000,
    },
    "gemma4:e4b": {
        "IT_High":    0.360916,
        "IT_Medium":  1.000000,
        "IT_Low":     0.533359,
        "PT_Medium":  1.000000,
        "EN_High":    0.001282,
        "EN_Medium":  1.000000,
        "VRAM_Low":   1.000000,
        "VRAM_Medium": 0.000000,
        "VRAM_High":  0.000000,
        "OSS_High":   0.250000,
        "OSS_Medium": 0.750000,
        "OSS_Low":    0.000000,
    },
    "mistral-small3.2:24b": {
        "IT_High":    0.802840,
        "IT_Medium":  1.000000,
        "IT_Low":     0.278588,
        "PT_Medium":  1.000000,
        "EN_High":    0.466677,
        "EN_Medium":  1.000000,
        "VRAM_Low":   0.000000,
        "VRAM_Medium": 1.000000,
        "VRAM_High":  0.000000,
        "OSS_High":   0.000000,
        "OSS_Medium": 1.000000,
        "OSS_Low":    0.000000,
    },
    "nemotron-cascade-2:30b": {
        "IT_High":    0.754401,
        "IT_Medium":  1.000000,
        "IT_Low":     0.277316,
        "PT_Medium":  0.829750,
        "EN_High":    0.702763,
        "EN_Medium":  0.849025,
        "VRAM_Low":   0.000000,
        "VRAM_Medium": 0.750000,
        "VRAM_High":  0.250000,
        "OSS_High":   0.250000,
        "OSS_Medium": 0.750000,
        "OSS_Low":    0.000000,
    },
}

# Maps the linguistic label key (e.g. "IT_Medium") to the MLN ground atom name
# (e.g. "it_medium") used in RULES antecedents.
# This translation step converts the rule condition vocabulary into MLN ground atoms.
_MEMBERSHIP_KEY_TO_ATOM: dict[str, str] = {
    "IT_High":     "it_high",     # "Italian IS High"      → ground atom it_high
    "IT_Medium":   "it_medium",   # "Italian IS Medium"    → ground atom it_medium
    "IT_Low":      "it_low",      # "Italian IS Low"       → ground atom it_low
    "PT_Medium":   "pt_medium",   # "Portuguese IS Medium" → ground atom pt_medium
    "EN_High":     "en_high",     # "English IS High"      → ground atom en_high
    "EN_Medium":   "en_medium",   # "English IS Medium"    → ground atom en_medium
    "VRAM_Low":    "vram_low",    # "Vram IS Low"          → ground atom vram_low
    "VRAM_Medium": "vram_medium", # "Vram IS Medium"       → ground atom vram_medium
    "VRAM_High":   "vram_high",   # "Vram IS High"         → ground atom vram_high
    "OSS_High":    "oss_high",    # "Oss IS High"          → ground atom oss_high
    "OSS_Medium":  "oss_medium",  # "Oss IS Medium"        → ground atom oss_medium
    "OSS_Low":     "oss_low",     # "Oss IS Low"           → ground atom oss_low
}

_INPUT_ATOMS  = sorted({a for rule in RULES for a in rule["antecedents"]})
_OUTPUT_ATOMS = sorted({rule["head"] for rule in RULES})
_ALL_ATOMS    = _INPUT_ATOMS + _OUTPUT_ATOMS  # fixed ordering for bit indexing

_EPS = 1e-9


def _log_odds(mu: float) -> float:
    """Convert membership value to log-odds for use as unary potential.

    (Pearl 1988; Richardson & Domingos 2006, §6)
    """
    mu = max(_EPS, min(1.0 - _EPS, mu))
    return math.log(mu / (1.0 - mu))


def _mln_inference(input_log_odds: dict[str, float]) -> dict[str, float]:
    """Exact MLN marginal inference via full world enumeration.

    (Richardson & Domingos 2006, §4)
    """
    n = len(_ALL_ATOMS)
    log_potentials: list[float] = []

    for bits in itertools.product((0, 1), repeat=n):
        world = dict(zip(_ALL_ATOMS, bits))
        lp = 0.0

        # Unary potentials from membership soft evidence  (Richardson & Domingos 2006, §6)
        for atom, lo in input_log_odds.items():
            lp += lo * world[atom]

        # Rule potentials: each IF-THEN rule is encoded as ¬A1 ∨ … ∨ ¬An ∨ C
        # (Richardson & Domingos 2006, §2).
        for rule in RULES:
            antecedents_true = all(world[a] for a in rule["antecedents"])  # AND conjunction of rule body atoms
            head_true = bool(world[rule["head"]])                          # truth of the THEN consequent atom
            if not (antecedents_true and not head_true):                   # rule ¬A1∨…∨¬An∨C is satisfied
                lp += rule["weight"]

        log_potentials.append(lp)

    max_lp = max(log_potentials)
    potentials = [math.exp(lp - max_lp) for lp in log_potentials]
    Z = sum(potentials)  # partition function  (Richardson & Domingos 2006, §2)

    marginals: dict[str, float] = {a: 0.0 for a in _OUTPUT_ATOMS}
    for bits, pot in zip(itertools.product((0, 1), repeat=n), potentials):
        world = dict(zip(_ALL_ATOMS, bits))
        for atom in _OUTPUT_ATOMS:
            if world[atom]:
                marginals[atom] += pot / Z

    return marginals


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def run() -> dict[str, dict]:
    head_to_consequent = {r["head"]: r["consequent"] for r in RULES}
    results: dict[str, dict] = {}

    for model, flat in MEMBERSHIPS.items():
        # Translate membership values to log-odds atoms for the rule antecedents
        input_log_odds = {
            atom: _log_odds(flat.get(mem_key, 0.0))
            for mem_key, atom in _MEMBERSHIP_KEY_TO_ATOM.items()
            if atom in _INPUT_ATOMS
        }

        marginals = _mln_inference(input_log_odds)

        score = sum(
            marginals[head] * EV_TABLE[head_to_consequent[head]]
            for head in _OUTPUT_ATOMS
        )

        results[model] = {"marginals": marginals, "score": score}

    total = sum(info["score"] for info in results.values())
    for info in results.values():
        info["normalized"] = info["score"] / total if total else 0.0

    return results


def main() -> None:
    results = run()

    print(f"MLN rule weight: {RULE_WEIGHT}")
    print(f"Input atoms:  {_INPUT_ATOMS}")
    print(f"Output atoms: {_OUTPUT_ATOMS}")

    for model, info in results.items():
        print(f"\n{model}")
        print(f"  score={info['score']:.6f}  normalized={info['normalized']:.6f}")
        for head in sorted(info["marginals"]):
            print(f"  P({head})={info['marginals'][head]:.6f}")

    ranking = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)

    print("\nRanking")
    print("rank | model | score | normalized")
    for rank, (model, info) in enumerate(ranking, start=1):
        print(f"{rank} | {model} | {info['score']:.6f} | {info['normalized']:.6f}")

    print("\nLaTeX final ranking table")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{MLN ranking for example 2 (rule weight = " + str(RULE_WEIGHT) + r").}")
    print(r"\label{tab:mln-example-2-ranking}")
    print(r"\begin{tabular}{r l r r}")
    print(r"\hline")
    print(r"Rank & Model & Score & Normalized \\")
    print(r"\hline")
    for rank, (model, info) in enumerate(ranking, start=1):
        print(f"{rank} & {_escape_latex(model)} & {info['score']:.3f} & {info['normalized']:.3f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
