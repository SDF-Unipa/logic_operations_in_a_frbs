"""
Sensitivity analysis for routine_example_2.

For each tolerance level in PERTURBATION_PCTS and each of N_RUNS runs:
  - perturb every model's mean scores by a uniform random draw in
    [-pct, +pct] of the original mean
  - re-run the fuzzy inference and ranking
  - compare the new ranking to the original via Kendall tau

Per-run JSON files are written to METADATA_DIR/pct_<label>/.
A standalone tolerances_summary.json collects the average tau per tolerance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from tqdm import tqdm

from topsis_fuzzy.config import DEFAULT_DIM_MAP
from topsis_fuzzy.data_loader import LANGUAGE_KEYS, convert_raw_data, load_raw_data
from compute_output_weight_integrals import compute_integrals
from compute_trapezoidal_memberships import (
    HARDCODED_TRAPEZOIDAL_MEMBERSHIP_FUNCTIONS,
    build_trapezoidal_dataset,
    compute_memberships,
)


# Perturbation levels tested: 1%, 2.5%, then 5%–50% in 5% steps.
PERTURBATION_PCTS: list[float] = [0.01, 0.025] + [round(p / 100, 10) for p in range(5, 55, 5)]
# 100 independent Monte Carlo runs per tolerance level for stable tau estimates.
N_RUNS = 100
# Fixed seed ensures reproducibility: run k uses seed GLOBAL_SEED + k.
GLOBAL_SEED = 42
DATA_PATH = Path(__file__).with_name("scores_simplified.json")
METADATA_DIR = Path(__file__).with_name("example_2_metadata")

RULES = [
    {
        "name": "r1",
        "text": "IF Italian IS Medium AND Portuguese IS Medium AND English IS Medium AND Vram IS Low THEN Weight IS High",
        "antecedents": ["IT_Medium", "PT_Medium", "EN_Medium", "VRAM_Low"],
        "consequent": "High",
    },
    {
        "name": "r2",
        "text": "IF Italian IS Medium AND Portuguese IS Medium AND English IS Medium AND Oss IS High THEN Weight IS Very High",
        "antecedents": ["IT_Medium", "PT_Medium", "EN_Medium", "OSS_High"],
        "consequent": "Very High",
    },
    {
        "name": "r3",
        "text": "IF Italian IS High AND Portuguese IS Medium AND English IS Medium THEN Weight IS High",
        "antecedents": ["IT_High", "PT_Medium", "EN_Medium"],
        "consequent": "High",
    },
    {
        "name": "r4",
        "text": "IF Italian IS Low AND Portuguese IS Medium AND English IS Medium AND Vram IS High AND Oss IS Low THEN Weight IS Very Low",
        "antecedents": ["IT_Low", "PT_Medium", "EN_Medium", "VRAM_High", "OSS_Low"],
        "consequent": "Very Low",
    },
    {
        "name": "r5",
        "text": "IF English IS High AND Oss IS Medium AND Vram IS Medium THEN Weight IS Low",
        "antecedents": ["EN_High", "OSS_Medium", "VRAM_Medium"],
        "consequent": "Low",
    },
]


def _product(values: list[float]) -> float:
    # Product T-norm: α = Π μ(antecedent_k).  Larsen (1980) rule-firing strength.
    result = 1.0
    for v in values:
        result *= v
    return result


def _flatten_memberships(
    memberships: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    # Reshapes memberships[dim][model][label] → flat[model]["DIM_Label"].
    flat: dict[str, dict[str, float]] = {}
    for dim, dim_data in memberships.items():
        for model, label_values in dim_data.items():
            flat.setdefault(model, {})
            for label, value in label_values.items():
                flat[model][f"{dim}_{label}"] = value
    return flat


def _build_output_sets(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        row["fuzzy_set"]: {
            "integral_mu_dt": float(row["integral_mu_dt"]),
            "integral_t_mu_dt": float(row["integral_t_mu_dt"]),
        }
        for row in rows
    }


def _infer_ranking(raw_data: dict) -> tuple[list[str], dict[str, dict]]:
    output_sets = _build_output_sets(compute_integrals())
    data = convert_raw_data(raw_data, DEFAULT_DIM_MAP)
    trapezoidal_data = build_trapezoidal_dataset(data)
    memberships = _flatten_memberships(
        compute_memberships(trapezoidal_data, HARDCODED_TRAPEZOIDAL_MEMBERSHIP_FUNCTIONS)
    )

    denominator = sum(output_sets[rule["consequent"]]["integral_mu_dt"] for rule in RULES)
    results: dict[str, dict] = {}

    for model, values in memberships.items():
        numerator = 0.0
        rule_outputs = []
        for rule in RULES:
            alpha = _product([values[key] for key in rule["antecedents"]])
            weighted_output = alpha * output_sets[rule["consequent"]]["integral_t_mu_dt"]
            numerator += weighted_output
            rule_outputs.append({
                "name": rule["name"],
                "alpha": alpha,
                "weighted_output": weighted_output,
                "consequent": rule["consequent"],
            })
        results[model] = {"rule_outputs": rule_outputs, "raw": numerator / denominator if denominator else 0.0}

    raw_total = sum(info["raw"] for info in results.values())
    for info in results.values():
        info["normalized"] = info["raw"] / raw_total if raw_total else 0.0

    ranking = [
        model for model, _ in sorted(results.items(), key=lambda x: x[1]["normalized"], reverse=True)
    ]
    return ranking, results


def _perturb_raw_data(
    raw_data: dict, pct: float, rng: np.random.Generator
) -> tuple[dict, dict]:
    # Applies a multiplicative perturbation to each model's mean score:
    #   new_mean = original_mean × (1 + δ),  δ ~ Uniform(−pct, +pct)
    # Standard deviations are kept unchanged; only the mean is shifted.
    # A fresh delta is drawn independently for each (model, dimension) pair.
    perturbed: dict = {}
    deltas: dict = {}

    for llm, llm_data in raw_data.items():
        perturbed[llm] = {}
        deltas[llm] = {}
        for dim_key, dim_data in llm_data.items():
            delta = float(rng.uniform(-pct, pct))
            if dim_key in LANGUAGE_KEYS:
                stats = dim_data["@10%"]
                new_mean = float(stats["mean"]) * (1.0 + delta)
                perturbed[llm][dim_key] = {"@10%": {"mean": new_mean, "std": float(stats["std"])}}
            else:
                new_mean = float(dim_data["mean"]) * (1.0 + delta)
                perturbed[llm][dim_key] = {"mean": new_mean, "std": float(dim_data["std"])}
            deltas[llm][dim_key] = delta

    return perturbed, deltas


def _ranking_order(ranking: list[str], all_models: list[str]) -> list[int]:
    # Converts a ranked list to a positional vector aligned with all_models
    # so that scipy.stats.kendalltau can compare two rankings as integer arrays.
    rank_of = {model: i for i, model in enumerate(ranking)}
    return [rank_of[m] for m in all_models]


def _results_to_serialisable(ranking: list[str], results: dict[str, dict]) -> list[dict]:
    return [
        {
            "rank": i + 1,
            "model": model,
            "raw": results[model]["raw"],
            "normalized": results[model]["normalized"],
        }
        for i, model in enumerate(ranking)
    ]


def _pct_label(pct: float) -> str:
    s = f"{pct * 100:g}".replace(".", "p")
    return f"pct_{s}"


def _run_batch(
    raw_data: dict,
    all_models: list[str],
    original_ranking: list[str],
    pct: float,
    batch_dir: Path,
) -> list[dict]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    run_summaries: list[dict] = []

    for run_idx in range(N_RUNS):
        seed = GLOBAL_SEED + run_idx
        rng = np.random.default_rng(seed)

        perturbed_data, deltas = _perturb_raw_data(raw_data, pct, rng)
        new_ranking, new_results = _infer_ranking(perturbed_data)

        tau_stat = kendalltau(
            _ranking_order(original_ranking, all_models),
            _ranking_order(new_ranking, all_models),
        )

        run_payload = {
            "run": run_idx + 1,
            "seed": seed,
            "perturbation_pct": pct,
            "deltas": deltas,
            "ranking": _results_to_serialisable(new_ranking, new_results),
            "kendall_tau": float(tau_stat.statistic),
            "kendall_pvalue": float(tau_stat.pvalue),
        }

        run_path = batch_dir / f"run_{run_idx + 1:02d}.json"
        run_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

        run_summaries.append({
            "run": run_idx + 1,
            "seed": seed,
            "kendall_tau": float(tau_stat.statistic),
            "kendall_pvalue": float(tau_stat.pvalue),
            "ranking": [entry["model"] for entry in run_payload["ranking"]],
        })

    taus = [r["kendall_tau"] for r in run_summaries]
    batch_summary = {
        "perturbation_pct": pct,
        "global_seed": GLOBAL_SEED,
        "n_runs": N_RUNS,
        "runs": run_summaries,
        "stats": {
            "mean_kendall_tau": float(np.mean(taus)),
            "std_kendall_tau": float(np.std(taus)),
            "min_kendall_tau": float(np.min(taus)),
            "max_kendall_tau": float(np.max(taus)),
        },
    }
    (batch_dir / "summary.json").write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")

    return run_summaries


def run() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_raw_data(DATA_PATH)
    all_models = list(raw_data.keys())

    original_ranking, original_results = _infer_ranking(raw_data)
    original_serialisable = _results_to_serialisable(original_ranking, original_results)

    original_path = METADATA_DIR / "original.json"
    original_path.write_text(
        json.dumps({"ranking": original_serialisable}, indent=2),
        encoding="utf-8",
    )
    print(f"original -> {original_path}\n")

    tolerances_rows: list[dict] = []

    for pct in tqdm(PERTURBATION_PCTS, desc="tolerances", unit="tol"):
        label = _pct_label(pct)
        batch_dir = METADATA_DIR / label
        run_summaries = _run_batch(raw_data, all_models, original_ranking, pct, batch_dir)

        taus = [r["kendall_tau"] for r in run_summaries]
        tolerances_rows.append({
            "perturbation_pct": pct,
            "label": label,
            "mean_kendall_tau": float(np.mean(taus)),
            "std_kendall_tau": float(np.std(taus)),
            "min_kendall_tau": float(np.min(taus)),
            "max_kendall_tau": float(np.max(taus)),
        })
        tqdm.write(f"  {pct*100:g}%  mean tau={np.mean(taus):.4f}  std={np.std(taus):.4f}")

    tolerances_summary = {
        "global_seed": GLOBAL_SEED,
        "n_runs": N_RUNS,
        "original_ranking": [entry["model"] for entry in original_serialisable],
        "tolerances": tolerances_rows,
    }
    tol_path = METADATA_DIR / "tolerances_summary.json"
    tol_path.write_text(json.dumps(tolerances_summary, indent=2), encoding="utf-8")
    print(f"\ntolerances summary -> {tol_path}")


if __name__ == "__main__":
    run()
