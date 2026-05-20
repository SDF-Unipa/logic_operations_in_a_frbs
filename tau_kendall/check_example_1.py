# Larsen inference for Example 1 with the product T-norm (reference baseline).
# T_product(a, b) = a · b  — the T-norm used by Larsen (1980).
# Identical logic to check_example_1.py in the root folder; duplicated here
# so that all T-norm variants can be compared in the same directory.
#
# Reference:
#   Larsen, P. M. (1980). Industrial applications of fuzzy logic control.
#   International Journal of Man-Machine Studies, 12(1), 3–10.
#   https://doi.org/10.1016/S0020-7373(80)80050-2
import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_WEIGHT_INTEGRALS_CSV = BASE_DIR / "output_weight_integrals.csv"
TRAPEZOIDAL_MEMBERSHIPS_CSV = BASE_DIR / "trapezoidal_memberships.csv"


def load_output_sets(path: Path) -> dict[str, dict[str, float]]:
    output_sets = {}

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output_sets[row["fuzzy_set"]] = {
                "integral_mu_dt": float(row["integral_mu_dt"]),
                "integral_t_mu_dt": float(row["integral_t_mu_dt"]),
            }

    return output_sets


def load_memberships(path: Path) -> dict[str, dict[str, float]]:
    memberships_by_label: dict[str, dict[str, dict[str, float]]] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        current_label = None
        headers = None

        for row in reader:
            if not row:
                current_label = None
                headers = None
                continue
            if len(row) == 1:
                current_label = row[0]
                memberships_by_label[current_label] = {}
                headers = None
                continue
            if row[0] == "Model":
                headers = row[1:]
                continue

            model = row[0]
            memberships_by_label[current_label][model] = {
                dimension: float(value)
                for dimension, value in zip(headers, row[1:])
            }

    memberships = {}

    for label, label_models in memberships_by_label.items():
        for model, dimension_values in label_models.items():
            memberships.setdefault(model, {})
            for dimension, value in dimension_values.items():
                memberships[model][f"{dimension}_{label}"] = value

    return memberships


def product(values: list[float]) -> float:
    # Product T-norm: T(a, b) = a · b.  Larsen (1980) rule-firing strength.
    result = 1.0
    for value in values:
        result *= value
    return result


def escape_latex(value: str) -> str:
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
    return "".join(replacements.get(char, char) for char in value)


OUTPUT_SETS = load_output_sets(OUTPUT_WEIGHT_INTEGRALS_CSV)
MEMBERSHIPS = load_memberships(TRAPEZOIDAL_MEMBERSHIPS_CSV)

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


DENOMINATOR = sum(OUTPUT_SETS[rule["consequent"]]["integral_mu_dt"] for rule in RULES)


results = {}

for model, values in MEMBERSHIPS.items():
    rule_outputs = []
    numerator = 0.0

    for rule in RULES:
        alpha = product([values[key] for key in rule["antecedents"]])
        weighted_output = alpha * OUTPUT_SETS[rule["consequent"]]["integral_t_mu_dt"]
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

    raw = numerator / DENOMINATOR if DENOMINATOR else 0.0
    results[model] = {
        "rule_outputs": rule_outputs,
        "raw": raw,
    }


raw_total = sum(info["raw"] for info in results.values())

for info in results.values():
    info["normalized"] = info["raw"] / raw_total if raw_total else 0.0


for model, info in results.items():
    print(model)
    print(f"  raw={info['raw']:.15f}")
    print(f"  normalized={info['normalized']:.15f}")
    for rule_output in info["rule_outputs"]:
        print(
            f"  {rule_output['name']} alpha={rule_output['alpha']:.15f} "
            f"weighted_output={rule_output['weighted_output']:.15f}"
        )


ranking = sorted(results.items(), key=lambda item: item[1]["normalized"], reverse=True)

print('\n')
print(25*'#')
print('All rules (reference)\n')

for rule in RULES:
    print(' * ',rule['text'])

print("\nRanking")
print("rank | model | raw | normalized")
for index, (model, info) in enumerate(ranking, start=1):
    print(f"{index} | {model} | {info['raw']:.15f} | {info['normalized']:.15f}")
    
print(25*'#')
print('\n')

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
        f"{index} & {escape_latex(model)} & "
        f"{info['raw']:.3f} & {info['normalized']:.3f} \\\\"
    )
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
