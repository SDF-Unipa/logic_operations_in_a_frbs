# Builds the trapezoidal input membership table and optional reference plots.
#
# Each model's score distribution is represented as a trapezoidal fuzzy number
#   trap(mean − 2σ, mean − 0.5σ, mean + 0.5σ, mean + 2σ)
# whose overlap with each linguistic MF (Low / Medium / High) is the
# max-min overlap degree:
#   μ(dim, label) = max_t  min(model_curve(t), mf_curve(t))
#
# This is the membership value used as the rule-firing strength in the
# Larsen fuzzy inference engine (Larsen 1980, §2).
#
# Reference:
#   Larsen, P. M. (1980). Industrial applications of fuzzy logic control.
#   International Journal of Man-Machine Studies, 12(1), 3–10.
#   https://doi.org/10.1016/S0020-7373(80)80050-2
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

from topsis_fuzzy.config import DEFAULT_DIM_MAP
from topsis_fuzzy.data_loader import convert_raw_data, load_raw_data
from topsis_fuzzy.membership_functions import eval_mf_on_universe, scalar_mf


# Fixed linguistic MFs for each input dimension.
# Params (a, b, c, d) were fitted to the score distributions in
# scores_simplified.json using the ±0.5σ / ±2σ convention:
#   Low    → right-shoulder  (a==b: full membership below b, falls off to d)
#   Medium → full trapezoid
#   High   → left-shoulder  (c==d: rises from a, full membership above c)
HARDCODED_TRAPEZOIDAL_MEMBERSHIP_FUNCTIONS: dict[str, dict[str, dict[str, tuple[float, float, float, float] | str]]] = {
    "EN": {
        "Low": {"type": "trap", "params": (2.823, 2.823, 7.565, 14.679)},
        "Medium": {"type": "trap", "params": (7.565, 14.679, 19.421, 26.535)},
        "High": {"type": "trap", "params": (19.421, 26.535, 31.277, 31.277)},
    },
    "IT": {
        "Low": {"type": "trap", "params": (4.899, 4.899, 7.982, 12.608)},
        "Medium": {"type": "trap", "params": (7.982, 12.608, 15.692, 20.318)},
        "High": {"type": "trap", "params": (15.692, 20.318, 23.401, 23.401)},
    },
    "PT": {
        "Low": {"type": "trap", "params": (1.2, 1.2, 5.792, 12.679)},
        "Medium": {"type": "trap", "params": (5.792, 12.679, 17.271, 24.158)},
        "High": {"type": "trap", "params": (17.271, 24.158, 28.75, 28.75)},
    },
    "VRAM": {
        "Low": {"type": "trap", "params": (8.0, 8.0, 16.0, 20.0)},
        "Medium": {"type": "trap", "params": (16.0, 20.0, 24.0, 28.0)},
        "High": {"type": "trap", "params": (24.0, 28.0, 32.0, 32.0)},
    },
    "OSS": {
        "Low": {"type": "trap", "params": (0.0, 0.0, 3.0, 5.0)},
        "Medium": {"type": "trap", "params": (3.0, 5.0, 7.0, 9.0)},
        "High": {"type": "trap", "params": (7.0, 9.0, 10.0, 10.0)},
    },
}


ROW_ORDER = tuple(DEFAULT_DIM_MAP.values())
LABEL_ORDER = ("Low", "Medium", "High")


def trapezoidal_fuzzy_number(mean: float, std: float) -> tuple[float, float, float, float]:
    # Encodes a score distribution as trap(mean−2σ, mean−0.5σ, mean+0.5σ, mean+2σ).
    # The plateau [mean−0.5σ, mean+0.5σ] captures the central 68% of scores;
    # the slopes extend to ±2σ to represent distribution uncertainty.
    return (mean - 2.0 * std, mean - 0.5 * std, mean + 0.5 * std, mean + 2.0 * std)


def build_trapezoidal_dataset(data: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, dict]]:
    trapezoidal_data: dict[str, dict[str, dict]] = {}

    for dim, dim_data in data.items():
        trapezoidal_data[dim] = {}
        for llm, stats in dim_data.items():
            trapezoidal_data[dim][llm] = {
                "mean": stats["mean"],
                "std": stats["std"],
                "trapezoidal": trapezoidal_fuzzy_number(stats["mean"], stats["std"]),
            }

    return trapezoidal_data


def make_universe(model_trap: tuple[float, float, float, float], mf_def: dict, points: int = 20001) -> np.ndarray:
    # Universe must span both the model trapezoid and the linguistic MF so that
    # the pointwise minimum is evaluated wherever either curve is nonzero.
    # A 5% margin prevents boundary artifacts from numerical integration.
    left = model_trap[0]
    right = model_trap[-1]

    if mf_def["type"] == "tri":
        a, _, c = mf_def["params"]
        left = min(left, a)
        right = max(right, c)
    elif mf_def["type"] == "trap":
        params = mf_def["params"]
        left = min(left, min(params))
        right = max(right, max(params))
    else:
        raise ValueError(f"Unsupported membership type: {mf_def['type']}")

    span = right - left
    margin = max(1e-6, 0.05 * max(span, 1e-6))
    return np.linspace(left - margin, right + margin, points)


def overlap_membership(model_trap: tuple[float, float, float, float], set_mf: dict, points: int = 20001) -> float:
    # Computes the max-min overlap degree:
    #   μ = max_t  min(model_curve(t), set_mf_curve(t))
    # This is the height of the intersection of two fuzzy sets, used as the
    # input membership value in the Larsen inference engine (Larsen 1980, §2).
    # Singleton models (std ≈ 0) degenerate to direct MF evaluation at the mean.
    a, b, c, d = model_trap
    is_singleton = np.isclose(a, b) and np.isclose(b, c) and np.isclose(c, d)

    if is_singleton:
        return float(scalar_mf(b, set_mf))

    universe = make_universe(model_trap, set_mf, points=points)
    model_curve = fuzz.trapmf(universe, model_trap)
    set_curve = eval_mf_on_universe(universe, set_mf)
    return float(np.max(np.minimum(model_curve, set_curve)))


def compute_memberships(
    trapezoidal_data: dict[str, dict[str, dict]],
    membership_functions: dict[str, dict[str, dict]],
) -> dict[str, dict[str, dict[str, float]]]:
    memberships: dict[str, dict[str, dict[str, float]]] = {}

    for dim, dim_data in trapezoidal_data.items():
        memberships[dim] = {}
        for llm, info in dim_data.items():
            memberships[dim][llm] = {}
            for label, set_mf in membership_functions[dim].items():
                memberships[dim][llm][label] = overlap_membership(info["trapezoidal"], set_mf)

    return memberships


def build_memberships(data_path: Path) -> tuple[list[str], dict[str, dict[str, dict]], dict[str, dict[str, dict[str, float]]]]:
    raw_data = load_raw_data(data_path)
    models = list(raw_data.keys())
    data = convert_raw_data(raw_data, DEFAULT_DIM_MAP)
    trapezoidal_data = build_trapezoidal_dataset(data)
    memberships = compute_memberships(trapezoidal_data, HARDCODED_TRAPEZOIDAL_MEMBERSHIP_FUNCTIONS)
    return models, trapezoidal_data, memberships


def write_memberships_csv(
    output_path: Path,
    models: list[str],
    memberships: dict[str, dict[str, dict[str, float]]],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for index, label in enumerate(LABEL_ORDER):
            writer.writerow([label])
            writer.writerow(["Model", *ROW_ORDER])
            for model in models:
                writer.writerow([model, *[memberships[score][model][label] for score in ROW_ORDER]])
            if index < len(LABEL_ORDER) - 1:
                writer.writerow([])


def write_membership_reference_images(
    output_dir: Path,
    models: list[str],
    trapezoidal_data: dict[str, dict[str, dict]],
    memberships: dict[str, dict[str, dict[str, float]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for score in ROW_ORDER:
        for model in models:
            trapezoid = trapezoidal_data[score][model]["trapezoidal"]
            mean_value = trapezoidal_data[score][model]["mean"]
            for label in LABEL_ORDER:
                membership_value = memberships[score][model][label]
                mf_def = HARDCODED_TRAPEZOIDAL_MEMBERSHIP_FUNCTIONS[score][label]
                universe = _universe_for_overlap_plot(trapezoid, mf_def)
                mf_curve = eval_mf_on_universe(universe, mf_def)
                model_curve = _curve_for_trapezoid(universe, trapezoid)
                minimum_curve = np.minimum(model_curve, mf_curve)

                figure, axis = plt.subplots(figsize=(8, 5))
                axis.plot(universe, model_curve, linewidth=2.0, label="trapezoidal input score")
                axis.plot(universe, mf_curve, linewidth=2.0, label=f"{label} membership function")
                axis.plot(universe, minimum_curve, linewidth=2.4, color="tab:green", label="minimum")
                axis.fill_between(universe, 0.0, minimum_curve, color="tab:green", alpha=0.12)
                axis.axhline(
                    membership_value,
                    linestyle=":",
                    linewidth=1.2,
                    color="tab:red",
                    label=f"membership value={membership_value:.3f}",
                )
                axis.axvline(
                    mean_value,
                    linestyle="--",
                    linewidth=1.0,
                    color="tab:orange",
                    alpha=0.6,
                    label=f"score={mean_value:.3f}",
                )
                axis.set_title(f"{score} | {model} | {label}")
                axis.set_xlabel("score")
                axis.set_ylabel("membership")
                axis.set_ylim(-0.05, 1.05)
                axis.grid(True, alpha=0.3)
                axis.legend(frameon=False, fontsize=8, loc="best")
                figure.tight_layout()
                figure.savefig(output_dir / f"{score.lower()}_{_slugify(model)}_{label.lower()}.png", dpi=160)
                plt.close(figure)


def _curve_for_trapezoid(universe: np.ndarray, trapezoid: tuple[float, float, float, float]) -> np.ndarray:
    if _is_singleton_trapezoid(trapezoid):
        curve = np.zeros_like(universe)
        curve[np.argmin(np.abs(universe - trapezoid[1]))] = 1.0
        return curve
    return fuzz.trapmf(universe, trapezoid)


def _is_singleton_trapezoid(trapezoid: tuple[float, float, float, float]) -> bool:
    return (
        np.isclose(trapezoid[0], trapezoid[1])
        and np.isclose(trapezoid[1], trapezoid[2])
        and np.isclose(trapezoid[2], trapezoid[3])
    )


def _universe_for_overlap_plot(trapezoid: tuple[float, float, float, float], mf_def: dict) -> np.ndarray:
    left = min(trapezoid[0], mf_def["params"][0])
    right = max(trapezoid[-1], mf_def["params"][-1])
    margin = max(0.1, 0.08 * max(right - left, 1.0))
    return np.linspace(left - margin, right + margin, 2401)


def _slugify(value: str) -> str:
    sanitized = "".join(character if character.isalnum() else "_" for character in value)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute input memberships using the hardcoded trapezoidal membership functions from memberships_figs/membership_values.jpg, write them to a CSV file grouped by Low, Medium, and High tables, and generate per-membership reference plots."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=Path(__file__).with_name("scores_simplified.json"),
        type=Path,
        help="Path to the input JSON file. Defaults to scores_simplified.json in this directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("trapezoidal_memberships.csv"),
        help="Path to the CSV output file. Defaults to trapezoidal_memberships.csv in this directory.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).with_name("memberships_ref_images"),
        help="Directory where the per-membership reference plots will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models, trapezoidal_data, memberships = build_memberships(args.data_path)
    write_memberships_csv(args.output, models, memberships)
    write_membership_reference_images(args.images_dir, models, trapezoidal_data, memberships)
    print(args.output)
    print(args.images_dir)


if __name__ == "__main__":
    main()
