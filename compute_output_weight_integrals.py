# Computes ∫μ(t)dt and ∫t·μ(t)dt for each output weight membership function.
# These integrals are the building blocks of Larsen centroid defuzzification:
#
#   score = Σ_r  α_r · ∫t·μ_r(t)dt  /  Σ_r  ∫μ_r(t)dt
#
# where α_r is the rule firing strength (product T-norm over antecedent memberships)
# and the denominator is constant across models because it depends only on the
# output MFs, not on the input data.  (Larsen 1980, §2)
#
# Reference:
#   Larsen, P. M. (1980). Industrial applications of fuzzy logic control.
#   International Journal of Man-Machine Studies, 12(1), 3–10.
#   https://doi.org/10.1016/S0020-7373(80)80050-2
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from topsis_fuzzy.membership_functions import eval_mf_on_universe


# Output weight linguistic MFs on the universe [0, 10].
# Shoulder trapezoids at the extremes model saturation:
#   Very Low  → left-shoulder  (peak at 0, falls off by 2.25)
#   Very High → right-shoulder (rises from 7.75, peaks at 10)
# Params are (a, b, c, d) for trapmf_scalar.
OUTPUT_WEIGHT_MEMBERSHIP_FUNCTIONS = {
    "Very Low": {"type": "trap", "params": (0.0, 0.0, 1.0, 2.25)},
    "Low": {"type": "trap", "params": (1.0, 2.25, 3.25, 4.5)},
    "Medium": {"type": "trap", "params": (3.25, 4.5, 5.5, 6.75)},
    "High": {"type": "trap", "params": (5.5, 6.75, 7.75, 9.0)},
    "Very High": {"type": "trap", "params": (7.75, 9.0, 10.0, 10.0)},
}

OUTPUT_UNIVERSE_MIN = 0.0
OUTPUT_UNIVERSE_MAX = 10.0
NUMERIC_STEPS = 10_000  # trapezoidal rule resolution; 10 000 steps gives <0.01% error


def compute_integrals() -> list[dict[str, float | str]]:
    # Numerically integrate each MF over the output universe.
    # ∫μ(t)dt  = area under the MF  (denominator contribution per rule)
    # ∫t·μ(t)dt = first moment       (numerator contribution when α_r = 1)
    # expected_value = first moment / area = centroid of the MF alone
    universe = np.linspace(OUTPUT_UNIVERSE_MIN, OUTPUT_UNIVERSE_MAX, NUMERIC_STEPS + 1)
    rows: list[dict[str, float | str]] = []

    for label, mf_def in OUTPUT_WEIGHT_MEMBERSHIP_FUNCTIONS.items():
        mu = eval_mf_on_universe(universe, mf_def)
        integral_mu_dt = float(np.trapezoid(mu, universe))
        integral_t_mu_dt = float(np.trapezoid(universe * mu, universe))
        rows.append(
            {
                "fuzzy_set": label,
                "type": mf_def["type"],
                "params": str(mf_def["params"]),
                "integral_mu_dt": integral_mu_dt,
                "integral_t_mu_dt": integral_t_mu_dt,
                "expected_value": integral_t_mu_dt / integral_mu_dt,
            }
        )

    return rows


def write_csv(output_path: Path, rows: list[dict[str, float | str]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fuzzy_set", "type", "params", "integral_mu_dt", "integral_t_mu_dt", "expected_value"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save the Larsen-like output-weight area integral, first moment, and expected value to CSV using the trapezoidal definitions from memberships_figs/output_weights.jpg."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("output_weight_integrals.csv"),
        help="Path to the CSV output file. Defaults to output_weight_integrals.csv in this directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compute_integrals()
    write_csv(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
