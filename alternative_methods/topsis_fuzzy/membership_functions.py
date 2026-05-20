from __future__ import annotations

import numpy as np
import skfuzzy as fuzz


def build_standard_membership_functions(means: list[float]) -> dict[str, dict]:
    # Trapezoidal MFs derived from the sample mean and std of model scores.
    # Boundaries follow the ±0.5σ / ±2σ / ±3σ convention used throughout:
    #   Low    shoulder covers (μ−3σ, μ−3σ, μ−2σ, μ−0.5σ)
    #   Medium plateau covers (μ−2σ, μ−0.5σ, μ+0.5σ, μ+2σ)
    #   High   shoulder covers (μ+0.5σ, μ+2σ, μ+3σ, μ+3σ)
    # The hardcoded MFs in compute_trapezoidal_memberships.py use the same
    # parameterization fitted to the actual score distributions.
    mean = float(np.mean(means))
    sigma = float(np.std(means))
    return {
        "Low": {"type": "trap", "params": (mean - 3.0 * sigma, mean - 3.0 * sigma, mean - 2.0 * sigma, mean - 0.5 * sigma)},
        "Medium": {"type": "trap", "params": (mean - 2.0 * sigma, mean - 0.5 * sigma, mean + 0.5 * sigma, mean + 2.0 * sigma)},
        "High": {"type": "trap", "params": (mean + 0.5 * sigma, mean + 2.0 * sigma, mean + 3.0 * sigma, mean + 3.0 * sigma)},
    }


def build_standard_plot_membership_functions(means: list[float]) -> dict[str, dict]:
    return build_standard_membership_functions(means)


def build_plot_input_membership_functions(data: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, dict]]:
    membership_functions: dict[str, dict[str, dict]] = {}

    for dim, dim_data in data.items():
        means = [entry["mean"] for entry in dim_data.values()]
        membership_functions[dim] = build_standard_plot_membership_functions(means)

    return membership_functions


def build_vram_membership_functions() -> dict[str, dict]:
    return {
        "Low": {"type": "trap", "params": (0, 0, 16, 16)},
        "Medium": {"type": "trap", "params": (16, 16, 24, 24)},
        "High": {"type": "trap", "params": (24, 24, 32, 32)},
    }


def build_oss_membership_functions() -> dict[str, dict]:
    return {
        "Low": {"type": "trap", "params": (5, 5, 5, 7.5)},
        "Medium": {"type": "trap", "params": (5, 7.5, 7.5, 10)},
        "High": {"type": "trap", "params": (7.5, 10, 10, 10)},
    }


def build_input_membership_functions(data: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, dict]]:
    membership_functions: dict[str, dict[str, dict]] = {}

    for dim, dim_data in data.items():
        means = [entry["mean"] for entry in dim_data.values()]
        membership_functions[dim] = build_standard_membership_functions(means)

    return membership_functions


def trimf_scalar(x: float, tri: tuple[float, float, float]) -> float:
    # Degenerate cases (singleton, left-shoulder, right-shoulder) are handled
    # before the general formula to avoid division by zero when a==b or b==c.
    a, b, c = tri
    if a > b or b > c:
        raise ValueError(f"Invalid triangle: {tri}")

    if np.isclose(a, b) and np.isclose(b, c):
        return 1.0 if np.isclose(x, a) else 0.0
    if np.isclose(a, b):
        if x <= b:
            return 1.0
        if x >= c:
            return 0.0
        return (c - x) / (c - b)
    if np.isclose(b, c):
        if x <= a:
            return 0.0
        if x >= b:
            return 1.0
        return (x - a) / (b - a)
    if x <= a or x >= c:
        return 0.0
    if np.isclose(x, b):
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapmf_scalar(x: float, trap: tuple[float, float, float, float]) -> float:
    # Handles all degenerate trapezoid forms before the general formula
    # to avoid division by zero when a==b (left shoulder) or c==d (right shoulder).
    # Left-shoulder:  a==b, plateau [b,c], right slope (c,d)
    # Right-shoulder: c==d, left slope (a,b), plateau [b,c]
    # General:        left slope (a,b), plateau [b,c], right slope (c,d)
    a, b, c, d = trap
    if np.isclose(a, b) and np.isclose(b, c) and np.isclose(c, d):
        return 1.0 if np.isclose(x, a) else 0.0
    if np.isclose(a, b) and np.isclose(c, d) and a <= d:
        return 1.0 if a <= x <= d else 0.0
    if np.isclose(a, b):
        if not c < d:
            raise ValueError(f"Invalid left-shoulder trapezoid: {trap}")
        if x <= c:
            return 1.0
        if x >= d:
            return 0.0
        return (d - x) / (d - c)
    if np.isclose(c, d):
        if not a < b:
            raise ValueError(f"Invalid right-shoulder trapezoid: {trap}")
        if x <= a:
            return 0.0
        if x >= b:
            return 1.0
        return (x - a) / (b - a)
    if not (a <= b <= c <= d):
        raise ValueError(f"Invalid trapezoid: {trap}")
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    if c < x < d:
        return (d - x) / (d - c)
    return 0.0


def scalar_mf(x: float, mf_def: dict) -> float:
    if mf_def["type"] == "tri":
        return float(trimf_scalar(x, mf_def["params"]))
    if mf_def["type"] == "trap":
        return float(trapmf_scalar(x, mf_def["params"]))
    raise ValueError(f"Unsupported membership type: {mf_def['type']}")


def eval_mf_on_universe(universe: np.ndarray, mf_def: dict) -> np.ndarray:
    if mf_def["type"] == "tri":
        return fuzz.trimf(universe, mf_def["params"])
    # Use scalar loop instead of skfuzzy.trapmf because our trapmf_scalar
    # correctly handles degenerate shoulder trapezoids that skfuzzy ignores.
    if mf_def["type"] == "trap":
        return np.asarray([trapmf_scalar(float(x), mf_def["params"]) for x in universe], dtype=float)
    raise ValueError(f"Unsupported membership type: {mf_def['type']}")


def mf_centroid_from_def(mf_def: dict) -> float:
    # Centroid = ∫t·μ(t)dt / ∫μ(t)dt — same formula used in Larsen (1980)
    # defuzzification, computed analytically for triangles and numerically for trapezia.
    if mf_def["type"] == "tri":
        a, b, c = mf_def["params"]
        return float((a + b + c) / 3.0)
    if mf_def["type"] == "trap":
        universe = np.linspace(mf_def["params"][0], mf_def["params"][3], 2001)
        memberships = fuzz.trapmf(universe, mf_def["params"])
        area = np.trapezoid(memberships, universe)
        if np.isclose(area, 0.0):
            raise ZeroDivisionError(f"Zero area trapezoid: {mf_def['params']}")
        return float(np.trapezoid(universe * memberships, universe) / area)
    raise ValueError(f"Unsupported membership type: {mf_def['type']}")


def tri_neg(tri: tuple[float, float, float]) -> tuple[float, float, float]:
    # Negation of a triangular fuzzy number: −(a,b,c) = (−c, −b, −a).
    a, b, c = tri
    return (-c, -b, -a)


def tri_scalar_mul(k: float, tri: tuple[float, float, float]) -> tuple[float, float, float]:
    # Scalar multiplication preserves the ordering constraint a ≤ b ≤ c
    # by reversing the tuple when k < 0.
    a, b, c = tri
    if k >= 0:
        return (k * a, k * b, k * c)
    return (k * c, k * b, k * a)
