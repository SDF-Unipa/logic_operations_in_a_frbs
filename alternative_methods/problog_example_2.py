# =============================================================================
# ProbLog (AND-only rules) — example 2
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
#       score = Σ_head  P(head) × EV(consequent(head))
# =============================================================================
from __future__ import annotations

from problog import get_evaluatable
from problog.program import PrologString


# Maps the linguistic label key (e.g. "IT_Medium") to the ProbLog atom name
# (e.g. "it_medium") used in probabilistic facts and rule bodies.
# This translation step converts the rule condition vocabulary into ProbLog atoms.
_ATOM: dict[str, str] = {
    "IT_High":     "it_high",     # "Italian IS High"      → probabilistic fact it_high
    "IT_Medium":   "it_medium",   # "Italian IS Medium"    → probabilistic fact it_medium
    "IT_Low":      "it_low",      # "Italian IS Low"       → probabilistic fact it_low
    "PT_Medium":   "pt_medium",   # "Portuguese IS Medium" → probabilistic fact pt_medium
    "EN_High":     "en_high",     # "English IS High"      → probabilistic fact en_high
    "EN_Medium":   "en_medium",   # "English IS Medium"    → probabilistic fact en_medium
    "VRAM_Low":    "vram_low",    # "Vram IS Low"          → probabilistic fact vram_low
    "VRAM_Medium": "vram_medium", # "Vram IS Medium"       → probabilistic fact vram_medium
    "VRAM_High":   "vram_high",   # "Vram IS High"         → probabilistic fact vram_high
    "OSS_High":    "oss_high",    # "Oss IS High"          → probabilistic fact oss_high
    "OSS_Medium":  "oss_medium",  # "Oss IS Medium"        → probabilistic fact oss_medium
    "OSS_Low":     "oss_low",     # "Oss IS Low"           → probabilistic fact oss_low
}

# ---------------------------------------------------------------------------
# Each rule encodes one IF-THEN clause from the original rule set.
# "antecedents" are the label keys for the AND-joined rule body conditions;
# "head" is the Prolog atom for the THEN consequent.
# ---------------------------------------------------------------------------
RULES = [
    {
        "name": "r1",
        "text": "IF Italian IS Medium AND Portuguese IS Medium AND English IS Medium AND Vram IS Low THEN Weight IS High",
        "antecedents": ["IT_Medium", "PT_Medium", "EN_Medium", "VRAM_Low"],  # rule body
        "consequent": "High",
        "head": "weight_high",
    },
    {
        "name": "r2",
        "text": "IF Italian IS Medium AND Portuguese IS Medium AND English IS Medium AND Oss IS High THEN Weight IS Very High",
        "antecedents": ["IT_Medium", "PT_Medium", "EN_Medium", "OSS_High"],  # rule body
        "consequent": "Very High",
        "head": "weight_very_high",
    },
    {
        "name": "r3",
        "text": "IF Italian IS High AND Portuguese IS Medium AND English IS Medium THEN Weight IS High",
        "antecedents": ["IT_High", "PT_Medium", "EN_Medium"],  # rule body
        "consequent": "High",
        "head": "weight_high",
    },
    {
        "name": "r4",
        "text": "IF Italian IS Low AND Portuguese IS Medium AND English IS Medium AND Vram IS High AND Oss IS Low THEN Weight IS Very Low",
        "antecedents": ["IT_Low", "PT_Medium", "EN_Medium", "VRAM_High", "OSS_Low"],  # rule body
        "consequent": "Very Low",
        "head": "weight_very_low",
    },
    {
        "name": "r5",
        "text": "IF English IS High AND Oss IS Medium AND Vram IS Medium THEN Weight IS Low",
        "antecedents": ["EN_High", "OSS_Medium", "VRAM_Medium"],  # rule body
        "consequent": "Low",
        "head": "weight_low",
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
#   VRAM Low (8.0, 8.0, 16.0, 20.0)  VRAM Med (16.0, 20.0, 24.0, 28.0)  VRAM High (24.0, 28.0, 32.0, 32.0)
#   OSS Low  (0.0, 0.0, 3.0, 5.0)    OSS Med  (3.0, 5.0, 7.0, 9.0)      OSS High  (7.0, 9.0, 10.0, 10.0)
# Source: compute_trapezoidal_memberships.build_memberships()
MEMBERSHIPS: dict[str, dict[str, float]] = {
    "MichelRosselli/apertus": {
        "IT_High":     0.114389,
        "IT_Medium":   0.913492,
        "IT_Low":      0.953937,
        "PT_Medium":   0.713399,
        "EN_High":     0.000000,
        "EN_Medium":   0.753163,
        "VRAM_Low":    1.000000,
        "VRAM_Medium": 0.000000,
        "VRAM_High":   0.000000,
        "OSS_High":    1.000000,
        "OSS_Medium":  0.000000,
        "OSS_Low":     0.000000,
    },
    "gemma4:e4b": {
        "IT_High":     0.360916,
        "IT_Medium":   1.000000,
        "IT_Low":      0.533359,
        "PT_Medium":   1.000000,
        "EN_High":     0.001282,
        "EN_Medium":   1.000000,
        "VRAM_Low":    1.000000,
        "VRAM_Medium": 0.000000,
        "VRAM_High":   0.000000,
        "OSS_High":    0.250000,
        "OSS_Medium":  0.750000,
        "OSS_Low":     0.000000,
    },
    "mistral-small3.2:24b": {
        "IT_High":     0.802840,
        "IT_Medium":   1.000000,
        "IT_Low":      0.278588,
        "PT_Medium":   1.000000,
        "EN_High":     0.466677,
        "EN_Medium":   1.000000,
        "VRAM_Low":    0.000000,
        "VRAM_Medium": 1.000000,
        "VRAM_High":   0.000000,
        "OSS_High":    0.000000,
        "OSS_Medium":  1.000000,
        "OSS_Low":     0.000000,
    },
    "nemotron-cascade-2:30b": {
        "IT_High":     0.754401,
        "IT_Medium":   1.000000,
        "IT_Low":      0.277316,
        "PT_Medium":   0.829750,
        "EN_High":     0.702763,
        "EN_Medium":   0.849025,
        "VRAM_Low":    0.000000,
        "VRAM_Medium": 0.750000,
        "VRAM_High":   0.250000,
        "OSS_High":    0.250000,
        "OSS_Medium":  0.750000,
        "OSS_Low":     0.000000,
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
    print(r"\caption{ProbLog ranking for example 2.}")
    print(r"\label{tab:problog-example-2-ranking}")
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
