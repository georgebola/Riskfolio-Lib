"""Configuration for the aggressive thematic portfolio analysis.

Centralizes the v4 weights, ticker universe, group definitions, Black-Litterman
forward views, and optimizer constraints so the rest of the package stays thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


CAPITAL: float = 7_500.0
RISK_FREE_RATE: float = 0.04
LOOKBACK_YEARS: int = 3

# v4 portfolio (the intuition-built baseline we benchmark against)
V4_WEIGHTS: Dict[str, float] = {
    "BOTZ": 0.22,
    "XAR":  0.14,
    "SHLD": 0.03,
    "RKLB": 0.05,
    "KTOS": 0.01,
    "URA":  0.14,
    "GRID": 0.09,
    "SMH":  0.05,
    "ARKQ": 0.04,
    "QTUM": 0.04,
    "CIBR": 0.05,
    "XBI":  0.05,
    "SYM":  0.02,
    "SGOV": 0.04,
    "BND":  0.03,
}

TICKERS: List[str] = list(V4_WEIGHTS.keys())

# Theme groupings for look-through industry exposure and group constraints.
# A ticker can belong to multiple themes (e.g. RKLB is aerospace + space).
THEMES: Dict[str, List[str]] = {
    "Robotics":      ["BOTZ", "ARKQ", "SYM"],
    "Aerospace":     ["XAR", "RKLB"],
    "Defense":       ["SHLD", "KTOS"],
    "Nuclear":       ["URA"],
    "Grid":          ["GRID"],
    "Semiconductors":["SMH"],
    "Quantum":       ["QTUM"],
    "Cybersecurity": ["CIBR"],
    "Biotech":       ["XBI"],
    "FixedIncome":   ["SGOV", "BND"],
}

# Forward views to feed Black-Litterman. These override naive historical means
# (which would extrapolate QTUM's 86% trailing return into the future).
# Two view types are supported:
#   - "absolute": expected annual return for a single ticker
#   - "relative": expected outperformance of group A vs group B
ABSOLUTE_VIEWS: Dict[str, float] = {
    "QTUM": 0.08,   # cap quantum extrapolation
    "XBI":  0.12,   # rate-cut beneficiary
    "BOTZ": 0.14,   # AI/robotics structural
    "ARKQ": 0.13,
    "SYM":  0.13,
    "URA":  0.13,   # nuclear demand
    "GRID": 0.12,
    "CIBR": 0.10,
    "SMH":  0.09,   # capped — already at ATH
}

# Aerospace outperforms Defense by 4% annually (long aerospace, short defense).
RELATIVE_VIEWS: List[Tuple[List[str], List[str], float]] = [
    (["XAR", "RKLB"], ["SHLD", "KTOS"], 0.04),
]

# Confidence in each view (higher = optimizer trusts BL view more vs prior).
# Riskfolio expects an Omega matrix; we expose a simple per-view confidence
# scalar in [0, 1] and translate to Omega in black_litterman.py.
VIEW_CONFIDENCE: float = 0.5


@dataclass
class Constraints:
    """Hard constraints applied to every optimization run."""

    long_only: bool = True
    weight_min: float = 0.0
    weight_max: float = 0.22  # BOTZ cap

    # Group-level bounds: each entry is (theme_name, lower, upper).
    group_bounds: List[Tuple[str, float, float]] = field(
        default_factory=lambda: [
            ("Robotics",   0.20, 0.25),
            ("Nuclear",    0.10, 0.20),  # standalone nuclear (URA only)
            ("FixedIncome",0.05, 0.20),
        ]
    )

    # Relative group constraints: lhs >= rhs + slack.
    # ("Aerospace", "Defense", 0.0) means aerospace weight >= defense weight.
    group_relative: List[Tuple[str, str, float]] = field(
        default_factory=lambda: [
            ("Aerospace", "Defense", 0.0),
        ]
    )


CONSTRAINTS = Constraints()
