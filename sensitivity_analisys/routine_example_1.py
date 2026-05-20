# Larsen fuzzy rule-based inference — Example 1 (positive Latin languages case).
#
# Programmatic version of check_example_1.py: computes MF integrals and
# membership values on-the-fly from scores_simplified.json instead of reading
# pre-generated CSVs, so that sensitivity_analysis_example_1.py can call
# run() on perturbed input data without touching the filesystem.
#
# Reference:
#   Larsen, P. M. (1980). Industrial applications of fuzzy logic control.
#   International Journal of Man-Machine Studies, 12(1), 3–10.
#   https://doi.org/10.1016/S0020-7373(80)80050-2
from __future__ import annotations

from pathlib import Path

from compute_output_weight_integrals import compute_integrals
from compute_trapezoidal_memberships import build_memberships


DATA_PATH = Path(__file__).with_name("scores_simplified.json")

RULES = [
    {
        "name": "r1",
        "text": "IF Italian IS High AND Portuguese IS High THEN Weight IS Very High",
        "antecedents": ["IT_High", "PT_High"],
        "consequent": "Very High",
    },
    {
        "name": "r2",
        "text": "IF Italian IS Medium AND Portuguese IS High THEN Weight IS High",
        "antecedents": ["IT_Medium", "PT_High"],
        "consequent": "High",
    },
    {
        "name": "r3",
        "text": "IF Italian IS Low AND Portuguese IS Low AND English IS High THEN Weight IS Medium",
        "antecedents": ["IT_Low", "PT_Low", "EN_High"],
        "consequent": "Medium",
    },
    {
        "name": "r4",
        "text": "IF Italian IS Low AND Portuguese IS High THEN Weight IS Very Low",
        "antecedents": ["IT_Low", "PT_High"],
        "consequent": "Very Low",
    },
    {
        "name": "r5",
        "text": "IF Italian IS High AND Portuguese IS Low THEN Weight IS Very Low",
        "antecedents": ["IT_High", "PT_Low"],
        "consequent": "Very Low",
    },
]


def _product(values: list[float]) -> float:
    # Product T-norm: α = Π μ(antecedent_k).  Larsen (1980) rule-firing strength.
    result = 1.0
    for v in values:
        result *= v
    return result


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def _build_output_sets(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        row["fuzzy_set"]: {
            "integral_mu_dt": float(row["integral_mu_dt"]),
            "integral_t_mu_dt": float(row["integral_t_mu_dt"]),
        }
        for row in rows
    }


def _flatten_memberships(
    memberships: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    # Reshapes memberships[dim][model][label] → flat[model]["DIM_Label"]
    # to match the key format expected by the RULES antecedents list.
    flat: dict[str, dict[str, float]] = {}
    for dim, dim_data in memberships.items():
        for model, label_values in dim_data.items():
            flat.setdefault(model, {})
            for label, value in label_values.items():
                flat[model][f"{dim}_{label}"] = value
    return flat


def run(data_path: Path = DATA_PATH) -> dict[str, dict]:
    output_sets = _build_output_sets(compute_integrals())
    _models, _trap_data, raw_memberships = build_memberships(data_path)
    memberships = _flatten_memberships(raw_memberships)

    denominator = sum(output_sets[rule["consequent"]]["integral_mu_dt"] for rule in RULES)
    results: dict[str, dict] = {}

    for model, values in memberships.items():
        rule_outputs = []
        numerator = 0.0

        for rule in RULES:
            alpha = _product([values[key] for key in rule["antecedents"]])
            weighted_output = alpha * output_sets[rule["consequent"]]["integral_t_mu_dt"]
            numerator += weighted_output
            rule_outputs.append(
                {
                    "name": rule["name"],
                    "text": rule["text"],
                    "alpha": alpha,
                    "weighted_output": weighted_output,
                    "consequent": rule["consequent"],
                }
            )

        raw = numerator / denominator if denominator else 0.0
        results[model] = {"rule_outputs": rule_outputs, "raw": raw}

    raw_total = sum(info["raw"] for info in results.values())
    for info in results.values():
        info["normalized"] = info["raw"] / raw_total if raw_total else 0.0

    return results


def main() -> None:
    results = run()

    for model, info in results.items():
        print(model)
        print(f"  raw={info['raw']:.15f}")
        print(f"  normalized={info['normalized']:.15f}")
        for ro in info["rule_outputs"]:
            print(
                f"  {ro['name']} alpha={ro['alpha']:.15f} "
                f"weighted_output={ro['weighted_output']:.15f}"
            )

    ranking = sorted(results.items(), key=lambda item: item[1]["normalized"], reverse=True)

    print("\nRanking")
    print("rank | model | raw | normalized")
    for index, (model, info) in enumerate(ranking, start=1):
        print(f"{index} | {model} | {info['raw']:.15f} | {info['normalized']:.15f}")

    print("\nLaTeX final ranking table")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Final ranking for the positive Latin languages case.}")
    print(r"\label{tab:positive-latin-languages-ranking}")
    print(r"\begin{tabular}{r l r r}")
    print(r"\hline")
    print(r"Rank & Model & Raw & Normalized \\")
    print(r"\hline")
    for index, (model, info) in enumerate(ranking, start=1):
        print(
            f"{index} & {_escape_latex(model)} & "
            f"{info['raw']:.3f} & {info['normalized']:.3f} \\\\"
        )
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
