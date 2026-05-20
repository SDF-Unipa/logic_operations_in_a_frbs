"""Print LaTeX ranking tables for all six method/example combinations.

Outputs each table to the console and writes them all to latex_tables.tex.

The six combinations are:
  MLN × {Example 1, Example 2}
  ProbLog × {Example 1, Example 2}
  TOPSIS × {Example 1, Example 2}
"""
from __future__ import annotations

import sys
import os

# Ensure sibling modules (mln_example_1, problog_example_2, …) are importable
# when the script is invoked from a different working directory.
sys.path.insert(0, os.path.dirname(__file__))

import mln_example_1
import mln_example_2
import problog_example_1
import problog_example_2
import topsis_example_1
import topsis_example_2


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def _table_score_normalized(
    results: dict,
    caption: str,
    label: str,
) -> list[str]:
    ranking = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{r l r r}",
        r"\hline",
        r"Rank & Model & Score & Normalized \\",
        r"\hline",
    ]
    for rank, (model, info) in enumerate(ranking, start=1):
        lines.append(
            rf"{rank} & {_escape_latex(model)} & {info['score']:.3f} & {info['normalized']:.3f} \\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return lines


def _table_topsis(
    results: dict,
    caption: str,
    label: str,
) -> list[str]:
    ranking = sorted(results.items(), key=lambda x: x[1]["topsis_score"], reverse=True)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{r l r}",
        r"\hline",
        r"Rank & Model & TOPSIS Score \\",
        r"\hline",
    ]
    for rank, (model, info) in enumerate(ranking, start=1):
        lines.append(
            rf"{rank} & {_escape_latex(model)} & {info['topsis_score']:.3f} \\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return lines


def main() -> None:
    sections: list[tuple[str, list[str]]] = [
        (
            "MLN — Example 1 (Latin languages)",
            _table_score_normalized(
                mln_example_1.run(),
                caption=f"MLN ranking for the positive Latin languages case (rule weight = {mln_example_1.RULE_WEIGHT}).",
                label="tab:mln-latin-languages-ranking",
            ),
        ),
        (
            "ProbLog — Example 1 (Latin languages)",
            _table_score_normalized(
                problog_example_1.run(),
                caption="ProbLog ranking for the positive Latin languages case.",
                label="tab:problog-latin-languages-ranking",
            ),
        ),
        (
            "TOPSIS — Example 1 (Latin languages)",
            _table_topsis(
                topsis_example_1.run(),
                caption="TOPSIS ranking for the positive Latin languages case (uniform weights, benefit: IT, PT, EN).",
                label="tab:topsis-latin-languages-ranking",
            ),
        ),
        (
            "MLN — Example 2 (full criteria)",
            _table_score_normalized(
                mln_example_2.run(),
                caption=f"MLN ranking for example 2 (rule weight = {mln_example_2.RULE_WEIGHT}).",
                label="tab:mln-example-2-ranking",
            ),
        ),
        (
            "ProbLog — Example 2 (full criteria)",
            _table_score_normalized(
                problog_example_2.run(),
                caption="ProbLog ranking for example 2.",
                label="tab:problog-example-2-ranking",
            ),
        ),
        (
            "TOPSIS — Example 2 (full criteria)",
            _table_topsis(
                topsis_example_2.run(),
                caption="TOPSIS ranking for example 2 (uniform weights, benefit: IT, PT, EN, OSS; cost: VRAM).",
                label="tab:topsis-example-2-ranking",
            ),
        ),
    ]

    all_lines: list[str] = []
    for title, table_lines in sections:
        header = f"% {'=' * 60}"
        print(header)
        print(f"% {title}")
        print(header)
        for line in table_lines:
            print(line)
        print()

        all_lines += [header, f"% {title}", header]
        all_lines += table_lines
        all_lines.append("")

    out_path = os.path.join(os.path.dirname(__file__), "latex_tables.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(all_lines))
    print(f"% Written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
