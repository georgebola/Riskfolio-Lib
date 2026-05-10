"""Build asset-level and group-level constraint matrices for Riskfolio.

Riskfolio expects constraints in the form A @ w >= B (after sign flipping).
We build the (A, B) matrix from a friendlier dataclass spec.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Constraints, THEMES


def _theme_indicator(theme: str, tickers: List[str]) -> np.ndarray:
    members = set(THEMES.get(theme, []))
    return np.array([1.0 if t in members else 0.0 for t in tickers])


def build_constraints(
    tickers: List[str],
    cfg: Constraints,
) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[float, float]]:
    """Return (A, B, (lb, ub)).

    - A, B encode group constraints in the form A @ w >= B (Riskfolio convention).
    - (lb, ub) are scalar lower/upper bounds applied to every asset.
    """
    rows_A: List[np.ndarray] = []
    rows_B: List[float] = []

    for theme, lo, hi in cfg.group_bounds:
        indicator = _theme_indicator(theme, tickers)
        if indicator.sum() == 0:
            continue
        # lower bound: indicator @ w >= lo
        rows_A.append(indicator)
        rows_B.append(lo)
        # upper bound: -indicator @ w >= -hi
        rows_A.append(-indicator)
        rows_B.append(-hi)

    for lhs, rhs, slack in cfg.group_relative:
        diff = _theme_indicator(lhs, tickers) - _theme_indicator(rhs, tickers)
        rows_A.append(diff)
        rows_B.append(slack)

    A = pd.DataFrame(np.vstack(rows_A) if rows_A else np.empty((0, len(tickers))),
                     columns=tickers)
    B = pd.DataFrame(np.array(rows_B).reshape(-1, 1) if rows_B else np.empty((0, 1)))

    bounds = (cfg.weight_min, cfg.weight_max)
    return A, B, bounds


def asset_classes_frame(tickers: List[str]) -> pd.DataFrame:
    """Map each ticker to its primary theme for reporting and group views."""
    primary: Dict[str, str] = {}
    for theme, members in THEMES.items():
        for t in members:
            primary.setdefault(t, theme)
    return pd.DataFrame({
        "Asset": tickers,
        "Theme": [primary.get(t, "Other") for t in tickers],
    })
