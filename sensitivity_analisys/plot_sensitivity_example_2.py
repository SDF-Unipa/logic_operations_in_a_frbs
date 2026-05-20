# Plots mean Kendall τ ± std vs. perturbation level for Example 2.
# Reads tolerances_summary.json produced by sensitivity_analysis_example_2.py.
# Kendall τ = 1 means the perturbed ranking is identical to the original;
# τ = 0 means no correlation; τ = −1 means fully reversed ranking.
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

SUMMARY_PATH = Path(__file__).with_name("example_2_metadata") / "tolerances_summary.json"
OUTPUT_PATH = Path(__file__).with_name("example_2_metadata") / "sensitivity_plot.png"


def main() -> None:
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    pcts = [row["perturbation_pct"] * 100 for row in data["tolerances"]]
    means = [row["mean_kendall_tau"] for row in data["tolerances"]]
    stds = [row["std_kendall_tau"] for row in data["tolerances"]]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        pcts, means, yerr=stds,
        marker="o", linewidth=1.8, markersize=5,
        capsize=4, capthick=1.2, elinewidth=1.0,
        color="steelblue", ecolor="steelblue", alpha=0.85,
        label="mean Kendall τ ± std",
    )

    ax.set_xlabel("Perturbation (%)", fontsize=12)
    ax.set_ylabel("Kendall τ", fontsize=12)
    ax.set_title("Robustness Analisys for Example 2", fontsize=12)
    ax.set_xticks(pcts)
    ax.set_xticklabels([f"{p:g}" for p in pcts], rotation=45, ha="right")
    ax.set_ylim(-0.3, 1.15)
    ax.axhline(1.0, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)
    ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)
    ax.axhline(-1.0, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
