from __future__ import annotations

from dataclasses import dataclass


# Maps JSON dimension keys to the short codes used throughout the codebase.
# Language keys ("English", "Italian", "Portuguese") are extracted from the
# nested "@10%" sub-dict in scores_simplified.json; the rest are top-level.
DEFAULT_DIM_MAP = {
    "English": "EN",
    "Italian": "IT",
    "Portuguese": "PT",
    "VRAM": "VRAM",
    "Open_Sourceness": "OSS",
}

DEFAULT_RULES = [
    "IF (IT is High AND EN is High AND PT is High) THEN W is High",
    "IF (OSS is High) THEN W is Very High",
]


# Immutable bundle of all parameters that define one inference scenario.
# Frozen to prevent accidental mutation during multi-run sensitivity loops.
@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    rules: list[str]
    output_universe_min: float
    output_universe_max: float
    output_universe_points: int
    output_membership_functions: dict[str, dict]
    normalization_mode: str
    aggregation_mode: str
    heading: str


# Output universe [0, 10]: the weight score is always non-negative.
# Shoulder trapezoids at the extremes (Very Low, Very High) encode
# saturation: any score ≤ 1 is fully "Very Low", any score ≥ 9 is fully "Very High".
POSITIVE_SCENARIO = ScenarioConfig(
    name="positive",
    rules=DEFAULT_RULES,
    output_universe_min=0.0,
    output_universe_max=10.0,
    output_universe_points=1001,
    output_membership_functions={
        "Very Low": {"type": "trap", "params": (0.0, 0.0, 1.0, 2.5)},
        "Low": {"type": "tri", "params": (1.5, 3.0, 4.5)},
        "Medium": {"type": "tri", "params": (3.5, 5.0, 6.5)},
        "High": {"type": "tri", "params": (5.5, 7.0, 8.5)},
        "Very High": {"type": "trap", "params": (7.5, 9.0, 10.0, 10.0)},
    },
    normalization_mode="sum",
    aggregation_mode="positive",
    heading="Positive Aggregation",
)


# Symmetric output universe [−3, 3] for signed aggregation scenarios
# where models can score negatively relative to a neutral baseline.
NEGATIVE_SCENARIO = ScenarioConfig(
    name="negative",
    rules=DEFAULT_RULES,
    output_universe_min=-3.0,
    output_universe_max=3.0,
    output_universe_points=1201,
    output_membership_functions={
        "Very Low": {"type": "tri", "params": (-3.0, -2.0, -1.0)},
        "Low": {"type": "tri", "params": (-2.0, -1.0, 0.0)},
        "Medium": {"type": "tri", "params": (-1.0, 0.0, 1.0)},
        "High": {"type": "tri", "params": (0.0, 1.0, 2.0)},
        "Very High": {"type": "tri", "params": (1.0, 2.0, 3.0)},
    },
    normalization_mode="shifted_sum",
    aggregation_mode="signed",
    heading="Negative Aggregation",
)
