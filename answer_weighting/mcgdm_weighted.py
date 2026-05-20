"""
MCGDM Weighted Formulation — mcgdm_weighted_formulation.md
Loads x_ij^(k) from results.json (produced by rate_computational.py)
and applies the two-step weighted aggregation:

  Step 1  x̄_ij = Σ_k  w_k · x_ij^(k)
  Step 2  S_i   = Σ_j  v_j · x̄_ij        (v_j = 1/3, uniform)
"""

import json

RESULTS_FILE = "results.json"

ALTERNATIVES = [
    ("A1", "answer.txt"),
    ("A2", "answer_bad.txt"),
    ("A3", "answer_bad_en.txt"),
]

CRITERIA = ["USEFULNESS", "FACTUAL CORRECTNESS", "TEXT QUALITY"]

MODEL_WEIGHTS = {
    "nemotron-cascade-2:30b": 0.548,
    "mistral-small3.2:24b":   0.301,
    "gemma4:e4b":             0.149,
    "MichelRosselli/apertus": 0.003,
}

V = 1 / len(CRITERIA)

# ── MCGDM formulation ─────────────────────────────────────────────────────────

def _score(results: dict, label: str, model: str, c: str) -> float:
    r = results[label]["ratings"][model]
    if "error" in r:
        return 0.0
    return r[c]["score"]


def aggregated_matrix(results: dict) -> dict:
    """Step 1 — x̄_ij = Σ_k  w_k · x_ij^(k)"""
    return {
        label: {
            c: round(sum(MODEL_WEIGHTS[m] * _score(results, label, m, c) for m in MODEL_WEIGHTS), 3)
            for c in CRITERIA
        }
        for label, _ in ALTERNATIVES
    }


def final_scores(x_bar: dict) -> dict:
    """Step 2 — S_i = Σ_j  v_j · x̄_ij"""
    return {
        label: round(sum(V * x_bar[label][c] for c in CRITERIA), 3)
        for label, _ in ALTERNATIVES
    }

# ── display ───────────────────────────────────────────────────────────────────

def _short(model: str) -> str:
    return model.split("/")[-1].split(":")[0]


def print_ratings_table(results: dict) -> None:
    col_m, col_s = 24, 20
    header = f"{'Expert':<{col_m}} {'':>4}" + "".join(f"{c:>{col_s}}" for c in CRITERIA)
    sep = "-" * len(header)
    print(f"\n{'─'*len(header)}\n Per-expert scores  x_ij^(k)\n{sep}")
    print(header + "\n" + sep)
    for model, wk in MODEL_WEIGHTS.items():
        tag = f"{_short(model)} (w={wk:.3f})"
        for idx, (label, _) in enumerate(ALTERNATIVES):
            prefix = f"{tag:<{col_m}}" if idx == 0 else f"{'':>{col_m}}"
            scores_str = "".join(f"{_score(results, label, model, c):>{col_s}}" for c in CRITERIA)
            print(f"{prefix} {label:>4}{scores_str}")
        print(sep)


def print_aggregated_matrix(x_bar: dict) -> None:
    col = 20
    header = f"{'':>4}" + "".join(f"{c:>{col}}" for c in CRITERIA)
    sep = "-" * len(header)
    print(f"\n{'─'*len(header)}\n Aggregated matrix  x̄_ij = Σ_k w_k · x_ij^(k)\n{sep}")
    print(header + "\n" + sep)
    for label, _ in ALTERNATIVES:
        print(f"{label:>4}" + "".join(f"{x_bar[label][c]:>{col}.3f}" for c in CRITERIA))
    print(sep)


def print_final_scores(scores: dict) -> None:
    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    fname = dict(ALTERNATIVES)
    print(f"\n{'─'*52}\n Final scores  S_i = Σ_j (1/3) · x̄_ij\n{'─'*52}")
    for rank, (label, score) in enumerate(ranking, 1):
        print(f"  {rank}.  {label}  {fname[label]:<26}  S = {score:.3f}")
    best_label, best_score = ranking[0]
    print(f"\n  A* = {best_label}  (\"{fname[best_label]}\")  [S = {best_score:.3f}]\n")

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    x_bar  = aggregated_matrix(results)
    scores = final_scores(x_bar)

    print_ratings_table(results)
    print_aggregated_matrix(x_bar)
    print_final_scores(scores)


if __name__ == "__main__":
    main()
