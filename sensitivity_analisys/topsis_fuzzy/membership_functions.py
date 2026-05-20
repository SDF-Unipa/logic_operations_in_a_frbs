from __future__ import annotations

import numpy as np
import skfuzzy as fuzz


def build_standard_membership_functions(means: list[float]) -> dict[str, dict]:
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


def trapmf_vector(x: np.ndarray, trap: tuple[float, float, float, float]) -> np.ndarray:
    # Vectorized version of trapmf_scalar using NumPy boolean masks.
    # Replaces the scalar loop in eval_mf_on_universe for performance during
    # sensitivity analysis, which calls this function N_RUNS × |PERTURBATION_PCTS|
    # times per run.  Handles all degenerate shoulder cases before the general formula.
    a, b, c, d = trap
    y = np.zeros(len(x), dtype=float)
    ab_eq = np.isclose(a, b)
    cd_eq = np.isclose(c, d)
    if ab_eq and np.isclose(b, c) and cd_eq:
        y[np.isclose(x, a)] = 1.0
        return y
    if ab_eq and cd_eq:
        y[(x >= a) & (x <= d)] = 1.0
        return y
    if ab_eq:
        y[x <= c] = 1.0
        mask = (x > c) & (x < d)
        y[mask] = (d - x[mask]) / (d - c)
        return y
    if cd_eq:
        y[x >= b] = 1.0
        mask = (x > a) & (x < b)
        y[mask] = (x[mask] - a) / (b - a)
        return y
    y[(x >= b) & (x <= c)] = 1.0
    mask = (x > a) & (x < b)
    y[mask] = (x[mask] - a) / (b - a)
    mask = (x > c) & (x < d)
    y[mask] = (d - x[mask]) / (d - c)
    return y


def eval_mf_on_universe(universe: np.ndarray, mf_def: dict) -> np.ndarray:
    if mf_def["type"] == "tri":
        return fuzz.trimf(universe, mf_def["params"])
    if mf_def["type"] == "trap":
        return trapmf_vector(universe, mf_def["params"])
    raise ValueError(f"Unsupported membership type: {mf_def['type']}")


def mf_centroid_from_def(mf_def: dict) -> float:
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
    a, b, c = tri
    return (-c, -b, -a)


def tri_scalar_mul(k: float, tri: tuple[float, float, float]) -> tuple[float, float, float]:
    a, b, c = tri
    if k >= 0:
        return (k * a, k * b, k * c)
    return (k * c, k * b, k * a)
