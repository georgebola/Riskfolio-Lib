"""Configuration for the aggressive thematic portfolio analysis.

Centralizes the current (v12) weights, ticker universe, group definitions,
Black-Litterman forward views, and optimizer constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


CAPITAL: float = 7_500.0
RISK_FREE_RATE: float = 0.04
LOOKBACK_YEARS: int = 3

# v12 — best version under final constraints (no KTOS/SMH/SYM/BND/ARKQ)
V4_WEIGHTS: Dict[str, float] = {
    "BOTZ": 0.20,
    "URA":  0.12,
    "CIBR": 0.15,
    "SHLD": 0.05,
    "XAR":  0.15,
    "RKLB": 0.05,
    "GRID": 0.08,
    "QTUM": 0.03,
    "XBI":  0.05,
    "SGOV": 0.12,
}

TICKERS: List[str] = list(V4_WEIGHTS.keys())

# Theme groupings for look-through industry exposure and group constraints.
THEMES: Dict[str, List[str]] = {
    "Robotics":      ["BOTZ"],
    "Aerospace":     ["XAR", "RKLB"],
    "Defense":       ["SHLD"],
    "Nuclear":       ["URA"],
    "Grid":          ["GRID"],
    "Quantum":       ["QTUM"],
    "Cybersecurity": ["CIBR"],
    "Biotech":       ["XBI"],
    "FixedIncome":   ["SGOV"],
}

# Forward views (Black-Litterman).
# Overrides naive historical means for speculative names.
ABSOLUTE_VIEWS: Dict[str, float] = {
    "QTUM": 0.08,   # cap quantum extrapolation — trailing 86% is not forward return
    "XBI":  0.12,   # rate-cut beneficiary, mean-revert after biotech bear
    "BOTZ": 0.16,   # AI/robotics infrastructure structural driver
    "URA":  0.15,   # nuclear demand: data-center power-purchase commitments
    "GRID": 0.13,   # grid modernisation tied to AI capex
    "CIBR": 0.12,   # cybersecurity: forced upgrade cycles, AI-accelerated threats
    "SHLD": 0.14,   # AI defense tech: autonomous systems, C2, drone swarms
    "XAR":  0.13,   # aerospace primes: NATO 2%+ rearmament, F-35 ramp
    "RKLB": 0.20,   # space launch: small-sat boom, hypersonic test contracts
}

# Aerospace outperforms Defense by 3% — enforces the structural gap.
RELATIVE_VIEWS: List[Tuple[List[str], List[str], float]] = [
    (["XAR", "RKLB"], ["SHLD"], 0.03),
]

# Higher confidence = BL view dominates the historical prior more strongly.
# 0.85 ensures QTUM cap sticks even against a strong trailing return.
VIEW_CONFIDENCE: float = 0.85


@dataclass
class Constraints:
    """Hard constraints applied to every optimization run."""

    long_only: bool = True
    weight_min: float = 0.0
    weight_max: float = 0.22  # BOTZ cap

    # Group-level bounds: (theme_name, lower, upper).
    group_bounds: List[Tuple[str, float, float]] = field(
        default_factory=lambda: [
            ("Robotics",      0.18, 0.22),  # BOTZ is the only robotics name
            ("Cybersecurity", 0.12, 0.20),  # CIBR should stay meaningful
            ("FixedIncome",   0.10, 0.18),  # SGOV cushion — don't let it drift below 10%
            ("Quantum",       0.00, 0.05),  # hard cap: QTUM never above 5%
            ("Biotech",       0.03, 0.08),  # XBI stays as a diversifier
            ("Defense",       0.03, 0.08),  # SHLD capped — not a core driver
        ]
    )

    # Relative group constraint: Aerospace >= Defense + slack.
    # Enforces a minimum 3pt gap so one drift cycle can't flip the relationship.
    group_relative: List[Tuple[str, str, float]] = field(
        default_factory=lambda: [
            ("Aerospace", "Defense", 0.03),
        ]
    )


CONSTRAINTS = Constraints()
