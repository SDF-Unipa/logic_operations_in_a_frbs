# Overlays Kendall τ curves for Example 1 and Example 2 on a single axes.
# Allows direct visual comparison of ranking robustness across the two rule sets.
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path(__file__).parent
SUMMARY_1 = BASE / "example_1_metadata" / "tolerances_summary.json"
SUMMARY_2 = BASE / "example_2_metadata" / "tolerances_summary.json"
OUTPUT_PATH = BASE / "sensitivity_plot_combined.png"


def _load(path: Path) -> tuple[list[float], list[float], list[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pcts  = [row["perturbation_pct"] * 100 for row in data["tolerances"]]
    means = [row["mean_kendall_tau"]       for row in data["tolerances"]]
    stds  = [row["std_kendall_tau"]        for row in data["tolerances"]]
    return pcts, means, stds


def main() -> None:
    pcts, means1, stds1 = _load(SUMMARY_1)
    _,    means2, stds2 = _load(SUMMARY_2)

    fig, ax = plt.subplots(figsize=(9, 5))

    common = dict(marker="o", linewidth=1.8, markersize=5,
                  capsize=4, capthick=1.2, elinewidth=1.0, alpha=0.85)

    ax.errorbar(pcts, means1, yerr=stds1,
                color="steelblue", ecolor="steelblue",
                label="Example 1", **common)
    ax.errorbar(pcts, means2, yerr=stds2,
                color="darkorange", ecolor="darkorange",
                label="Example 2", **common)

    ax.set_xlabel("Perturbation (%)", fontsize=12)
    ax.set_ylabel("Kendall τ", fontsize=12)
    ax.set_title("Robustness Analisys — Example 1 vs Example 2", fontsize=12)
    ax.set_xticks(pcts)
    ax.set_xticklabels([f"{p:g}" for p in pcts], rotation=45, ha="right")
    ax.set_ylim(-0.3, 1.15)
    ax.axhline(1.0, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)
    ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
