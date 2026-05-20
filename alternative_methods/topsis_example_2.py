# =============================================================================
# TOPSIS — example 2
#
# LITERATURE
# ----------
# Primary method:
#   Hwang, C. L., & Yoon, K. (1981). Multiple attribute decision making:
#   Methods and applications — A state-of-the-art survey.
#   Lecture Notes in Economics and Mathematical Systems, vol. 186.
#   Springer-Verlag. https://doi.org/10.1007/978-3-642-48318-9
#
# Python library:
#   Kizielewicz, B., Shekhovtsov, A., & Sałabun, W. (2023).
#   pymcdm — The universal library for solving multi-criteria
#   decision-making problems. SoftwareX, 22, 101368.
#   https://doi.org/10.1016/j.softx.2023.101368
#
# PARAMETER MAPPING
# -----------------
# Decision matrix  (Hwang & Yoon 1981, §3.1 "Step 1"):
#   Rows = LLM alternatives; columns = primal dimension scores IT, PT, EN, OSS, VRAM.
#   Each cell is the raw mean score for that model on that dimension,
#   taken directly from scores_simplified.json.
#
# Criterion types  (Hwang & Yoon 1981, §3.1 "Step 4"):
#   IT, PT, EN, OSS → benefit (+1): higher is preferable.
#   VRAM            → cost    (−1): lower VRAM required is preferable.
#
# Criterion weights  (Hwang & Yoon 1981, §3.1 "Step 3"):
#   Uniform: w = 1/5 for each criterion.
#
# Ideal / anti-ideal and closeness coefficient  (Hwang & Yoon 1981, §3.1 "Steps 4–7").
# =============================================================================
from __future__ import annotations

import numpy as np
from pymcdm.methods import TOPSIS


# ---------------------------------------------------------------------------
# Each criterion is one primal dimension score.
# IT, PT, EN, OSS → benefit; VRAM → cost (lower resource use is better).
# ---------------------------------------------------------------------------
CRITERIA = [
    {"name": "IT",   "description": "Italian score",          "type":  1},  # benefit
    {"name": "PT",   "description": "Portuguese score",       "type":  1},  # benefit
    {"name": "EN",   "description": "English score",          "type":  1},  # benefit
    {"name": "OSS",  "description": "Open-source score",      "type":  1},  # benefit: higher open-source degree is better
    {"name": "VRAM", "description": "VRAM required (GB)",     "type": -1},  # cost: lower VRAM required is better
]

# Raw mean scores per model per dimension.
# Source: scores_simplified.json  (@10% mean for IT, PT, EN; direct mean for OSS, VRAM)
SCORES: dict[str, dict[str, float]] = {
    "MichelRosselli/apertus": {"IT": 10.1, "PT":  9.2, "EN": 10.8, "OSS": 10.0, "VRAM":  9.5},
    "gemma4:e4b":             {"IT": 13.4, "PT": 14.7, "EN": 16.8, "OSS":  7.5, "VRAM": 11.2},
    "mistral-small3.2:24b":   {"IT": 16.7, "PT": 15.6, "EN": 18.4, "OSS":  5.0, "VRAM": 20.6},
    "nemotron-cascade-2:30b": {"IT": 16.4, "PT": 20.4, "EN": 22.2, "OSS":  7.5, "VRAM": 25.0},
}


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def run() -> dict[str, dict]:
    models = list(SCORES.keys())
    criteria_names = [c["name"] for c in CRITERIA]

    # Decision matrix: rows = models, columns = dimensions  (Hwang & Yoon 1981, §3.1 "Step 1")
    matrix = np.array(
        [[SCORES[model][c] for c in criteria_names] for model in models],
        dtype=float,
    )

    # Uniform weights: each criterion contributes equally  (Hwang & Yoon 1981, §3.1 "Step 3")
    weights = np.full(len(CRITERIA), 1.0 / len(CRITERIA))

    # Criterion types: benefit for IT/PT/EN/OSS, cost for VRAM  (Hwang & Yoon 1981, §3.1 "Step 4")
    types = np.array([c["type"] for c in CRITERIA], dtype=int)

    topsis_scores = TOPSIS()(matrix, weights, types)  # closeness coefficient s_iw  (Hwang & Yoon 1981, §3.1 "Steps 5–7")

    results: dict[str, dict] = {}
    for model, score in zip(models, topsis_scores):
        results[model] = {
            "scores": {c: SCORES[model][c] for c in criteria_names},
            "topsis_score": float(score),
        }

    return results


def main() -> None:
    results = run()

    criteria_names = [c["name"] for c in CRITERIA]
    print("Criteria:", criteria_names)
    print("Types:", {c["name"]: ("benefit" if c["type"] == 1 else "cost") for c in CRITERIA})
    print(f"Weights: uniform = {1.0 / len(CRITERIA):.4f} each\n")

    for model, info in results.items():
        scores_str = "  ".join(f"{c}={info['scores'][c]:.1f}" for c in criteria_names)
        print(f"{model}  topsis={info['topsis_score']:.6f}  [{scores_str}]")

    ranking = sorted(results.items(), key=lambda x: x[1]["topsis_score"], reverse=True)

    print("\nRanking")
    print("rank | model | topsis_score")
    for rank, (model, info) in enumerate(ranking, start=1):
        print(f"{rank} | {model} | {info['topsis_score']:.6f}")

    print("\nLaTeX final ranking table")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{TOPSIS ranking for example 2 (uniform weights, benefit: IT, PT, EN, OSS; cost: VRAM).}")
    print(r"\label{tab:topsis-example-2-ranking}")
    print(r"\begin{tabular}{r l r}")
    print(r"\hline")
    print(r"Rank & Model & TOPSIS Score \\")
    print(r"\hline")
    for rank, (model, info) in enumerate(ranking, start=1):
        print(f"{rank} & {_escape_latex(model)} & {info['topsis_score']:.3f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
