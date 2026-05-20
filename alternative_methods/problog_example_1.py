# =============================================================================
# ProbLog (AND-only rules) — example 1 (positive Latin languages case)
#
# LITERATURE
# ----------
# Original ProbLog language:
#   De Raedt, L., Kimmig, A., & Toivonen, H. (2007).
#   ProbLog: A probabilistic Prolog and its application in link discovery.
#   Proceedings of IJCAI-07, pp. 2468–2473.
#   https://lirias.kuleuven.be/handle/123456789/146072
#
# Inference via weighted Boolean formulas:
#   Fierens, D., Van den Broeck, G., Renkens, J., Shterionov, D.,
#   Gutmann, B., Thon, I., Janssens, G., & De Raedt, L. (2015).
#   Inference and learning in probabilistic logic programs using
#   weighted Boolean formulas.
#   Theory and Practice of Logic Programming, 15(3), 358–401.
#   https://doi.org/10.1017/S1471068414000076
#
# PARAMETER MAPPING
# -----------------
# Distribution semantics  (De Raedt et al. 2007, §2; Fierens et al. 2015, §2):
#   A ProbLog program defines a probability distribution over possible worlds.
#   Each world L is obtained by independently including or excluding each
#   probabilistic fact pi::Ai.  The probability of a world is:
#       P(L) = Π_{Ai ∈ L} pi  ×  Π_{Ai ∉ L} (1 − pi)
#   The probability of a query q is:
#       P(q) = Σ_{L ⊨ q} P(L)
#
# Probabilistic facts — membership values as P(atom):
#   (De Raedt et al. 2007, §2)
#
# Deterministic rules:
#   (Fierens et al. 2015, §3.1)
#
# Multiple rules sharing the same head:
#   (De Raedt et al. 2007, §2)
#
# Score aggregation — expected value:
#   The probability vector {P(weight_very_high), P(weight_high), …} is
#   collapsed into a scalar:
#       score = Σ_head  P(head) × EV(consequent(head))
# =============================================================================
from __future__ import annotations

from problog import get_evaluatable
from problog.program import PrologString


# Maps the linguistic label key (e.g. "IT_High") to the ProbLog atom name
# (e.g. "it_high") used in probabilistic facts and rule bodies.
# This translation step converts the rule condition vocabulary into ProbLog atoms.
_ATOM: dict[str, str] = {
    "IT_High":    "it_high",    # "Italian IS High"      → probabilistic fact it_high
    "IT_Medium":  "it_medium",  # "Italian IS Medium"    → probabilistic fact it_medium
    "IT_Low":     "it_low",     # "Italian IS Low"       → probabilistic fact it_low
    "PT_High":    "pt_high",    # "Portuguese IS High"   → probabilistic fact pt_high
    "PT_Medium":  "pt_medium",  # "Portuguese IS Medium" → probabilistic fact pt_medium
    "PT_Low":     "pt_low",     # "Portuguese IS Low"    → probabilistic fact pt_low
    "EN_High":    "en_high",    # "English IS High"      → probabilistic fact en_high
    "EN_Medium":  "en_medium",  # "English IS Medium"    → probabilistic fact en_medium
    "EN_Low":     "en_low",     # "English IS Low"       → probabilistic fact en_low
}

# ---------------------------------------------------------------------------
# Each rule encodes one IF-THEN clause from the original rule set.
# "antecedents" are the label keys for the AND-joined conditions in the rule
# body; "head" is the Prolog atom for the THEN consequent.
# The consequent string links back to EV_TABLE for score defuzzification.
# ---------------------------------------------------------------------------
RULES = [
    {
        "name": "r1",
        "text": "IF Italian IS High AND Portuguese IS High THEN Weight IS Very High",
        "antecedents": ["IT_High", "PT_High"],    # rule body: it_high, pt_high
        "consequent": "Very High",                # rule head label → EV_TABLE key
        "head": "weight_very_high",               # rule head atom name
    },
    {
        "name": "r2",
        "text": "IF Italian IS Medium AND Portuguese IS High THEN Weight IS High",
        "antecedents": ["IT_Medium", "PT_High"],  # rule body: it_medium, pt_high
        "consequent": "High",
        "head": "weight_high",
    },
    {
        "name": "r3",
        "text": "IF Italian IS Low AND Portuguese IS Low AND English IS High THEN Weight IS Medium",
        "antecedents": ["IT_Low", "PT_Low", "EN_High"],  # rule body: it_low, pt_low, en_high
        "consequent": "Medium",
        "head": "weight_medium",
    },
    {
        "name": "r4",
        "text": "IF Italian IS Low AND Portuguese IS High THEN Weight IS Very Low",
        "antecedents": ["IT_Low", "PT_High"],  # rule body: it_low, pt_high
        "consequent": "Very Low",
        "head": "weight_very_low",
    },
    {
        "name": "r5",
        "text": "IF Italian IS High AND Portuguese IS Low THEN Weight IS Very Low",
        "antecedents": ["IT_High", "PT_Low"],  # rule body: it_high, pt_low
        "consequent": "Very Low",
        "head": "weight_very_low",
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

# Input membership values per model — these become the probabilities of the
# probabilistic facts  (De Raedt et al. 2007, §2).
# Each value: max-min overlap degree between the model's trapezoidal score
# distribution trap(mean±2σ, mean±0.5σ) and the linguistic MF for that label.
# Input MF params (a,b,c,d) used to compute these values:
#   EN Low  (2.823, 2.823, 7.565, 14.679)  EN Med (7.565, 14.679, 19.421, 26.535)  EN High (19.421, 26.535, 31.277, 31.277)
#   IT Low  (4.899, 4.899, 7.982, 12.608)  IT Med (7.982, 12.608, 15.692, 20.318)  IT High (15.692, 20.318, 23.401, 23.401)
#   PT Low  (1.200, 1.200, 5.792, 12.679)  PT Med (5.792, 12.679, 17.271, 24.158)  PT High (17.271, 24.158, 28.750, 28.750)
# Source: compute_trapezoidal_memberships.build_memberships()
MEMBERSHIPS: dict[str, dict[str, float]] = {
    "MichelRosselli/apertus": {
        "IT_High":   0.114389,
        "IT_Medium": 0.913492,
        "IT_Low":    0.953937,
        "PT_High":   0.000000,
        "PT_Medium": 0.713399,
        "PT_Low":    0.721084,
        "EN_High":   0.000000,
        "EN_Medium": 0.753163,
        "EN_Low":    0.812977,
    },
    "gemma4:e4b": {
        "IT_High":   0.360916,
        "IT_Medium": 1.000000,
        "IT_Low":    0.533359,
        "PT_High":   0.253956,
        "PT_Medium": 1.000000,
        "PT_Low":    0.304500,
        "EN_High":   0.001282,
        "EN_Medium": 1.000000,
        "EN_Low":    0.056271,
    },
    "mistral-small3.2:24b": {
        "IT_High":   0.802840,
        "IT_Medium": 1.000000,
        "IT_Low":    0.278588,
        "PT_High":   0.387917,
        "PT_Medium": 1.000000,
        "PT_Low":    0.278964,
        "EN_High":   0.466677,
        "EN_Medium": 1.000000,
        "EN_Low":    0.244053,
    },
    "nemotron-cascade-2:30b": {
        "IT_High":   0.754401,
        "IT_Medium": 1.000000,
        "IT_Low":    0.277316,
        "PT_High":   0.771382,
        "PT_Medium": 0.829750,
        "PT_Low":    0.000000,
        "EN_High":   0.702763,
        "EN_Medium": 0.849025,
        "EN_Low":    0.000000,
    },
}

_HEADS = sorted({r["head"] for r in RULES})


def _build_program(flat: dict[str, float]) -> str:
    """Build the ProbLog program string for one model.

    Structure: probabilistic facts → deterministic rules → queries.
    (De Raedt et al. 2007, §2; Fierens et al. 2015, §3.1)
    """
    lines: list[str] = []

    # Probabilistic facts: each antecedent atom declared with P = membership value.
    # μ(dim, label) is used directly as the probability of the corresponding atom
    # being true, independently of all other atoms.  (De Raedt et al. 2007, §2)
    needed = {key for rule in RULES for key in rule["antecedents"]}
    for key in sorted(needed):
        prob = max(0.0, min(1.0, flat.get(key, 0.0)))
        lines.append(f"{prob:.10f}::{_ATOM[key]}.")  # p::atom  →  P(atom = true) = p

    lines.append("")

    # Deterministic rules: each IF-THEN clause becomes a Prolog rule with certainty 1.
    # The AND-joined rule body antecedents map to the comma-separated Prolog body;
    # the THEN consequent atom is the rule head.  (Fierens et al. 2015, §3.1)
    for rule in RULES:
        body = ", ".join(_ATOM[k] for k in rule["antecedents"])  # A1, A2, … (AND conjunction)
        lines.append(f"{rule['head']} :- {body}.")               # head :- body

    lines.append("")

    # Queries: one per distinct head atom
    for head in _HEADS:
        lines.append(f"query({head}).")

    return "\n".join(lines)


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
        program_text = _build_program(flat)
        program = PrologString(program_text)
        query_result = get_evaluatable().create_from(program).evaluate()

        probs: dict[str, float] = {str(term): float(prob) for term, prob in query_result.items()}

        score = sum(
            probs.get(head, 0.0) * EV_TABLE[head_to_consequent[head]]
            for head in _HEADS
        )

        results[model] = {"program": program_text, "probs": probs, "score": score}

    total = sum(info["score"] for info in results.values())
    for info in results.values():
        info["normalized"] = info["score"] / total if total else 0.0

    return results


def main() -> None:
    results = run()

    for model, info in results.items():
        print(f"\n{model}")
        print(f"  score={info['score']:.6f}  normalized={info['normalized']:.6f}")
        for head, prob in sorted(info["probs"].items()):
            print(f"  P({head})={prob:.6f}")

    ranking = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)

    print("\nRanking")
    print("rank | model | score | normalized")
    for rank, (model, info) in enumerate(ranking, start=1):
        print(f"{rank} | {model} | {info['score']:.6f} | {info['normalized']:.6f}")

    print("\nLaTeX final ranking table")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{ProbLog ranking for the positive Latin languages case.}")
    print(r"\label{tab:problog-latin-languages-ranking}")
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
