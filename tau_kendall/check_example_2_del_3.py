# Larsen inference for Example 2 with rule R3 removed.
# Rule R3: IF IT IS High AND PT IS Medium AND EN IS Medium THEN Weight IS High.
# Removing one rule at a time measures structural sensitivity to each rule's presence.
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
    result = 1.0
    for value in values:
        result *= value
    return result


OUTPUT_SETS = load_output_sets(OUTPUT_WEIGHT_INTEGRALS_CSV)
MEMBERSHIPS = load_memberships(TRAPEZOIDAL_MEMBERSHIPS_CSV)

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
    #{
    #    "name": "r3",
    #    "text": "IF Italian IS High AND Portuguese IS Medium AND English IS Medium THEN Weight IS High",
    #    "antecedents": ["IT_High", "PT_Medium", "EN_Medium"],
    #    "consequent": "High",
    #},
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
print('Deleted rule 3\n')

for rule in RULES:
    print(' * ',rule['text'])

print("\nRanking")
print("rank | model | raw | normalized")
for index, (model, info) in enumerate(ranking, start=1):
    print(f"{index} | {model} | {info['raw']:.15f} | {info['normalized']:.15f}")
    
print(25*'#')
print('\n')
