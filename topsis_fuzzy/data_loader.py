from __future__ import annotations

import json
from pathlib import Path


LANGUAGE_KEYS = ("English", "Italian", "Portuguese")


def load_raw_data(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def convert_raw_data(raw_data: dict, dim_map: dict[str, str]) -> dict[str, dict[str, dict[str, float]]]:
    data = {dim: {} for dim in dim_map.values()}

    for llm, llm_data in raw_data.items():
        for json_key, short_key in dim_map.items():
            stats = llm_data[json_key]["@10%"] if json_key in LANGUAGE_KEYS else llm_data[json_key]
            data[short_key][llm] = {
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
            }

    return data
