"""Black-Litterman posterior returns from forward views.

We implement the standard He-Litterman (1999) formulation:

    mu_bl = [(tau * Sigma)^-1 + P^T Omega^-1 P]^-1
            [(tau * Sigma)^-1 pi + P^T Omega^-1 Q]

    Sigma_bl = Sigma + [(tau * Sigma)^-1 + P^T Omega^-1 P]^-1

Where:
    pi    = implied equilibrium returns (we use historical mean as a proxy
            since we don't have market caps for thematic ETFs)
    P     = view selection matrix (n_views x n_assets)
    Q     = view return vector
    Omega = view confidence covariance (diagonal, scaled by view variance)
    tau   = scalar reflecting uncertainty in the prior (default 0.05)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import (
    ABSOLUTE_VIEWS,
    RELATIVE_VIEWS,
    VIEW_CONFIDENCE,
)


def build_views(tickers: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the (P, Q) view matrices from config."""
    rows: List[np.ndarray] = []
    q: List[float] = []

    for ticker, view in ABSOLUTE_VIEWS.items():
        if ticker not in tickers:
            continue
        row = np.zeros(len(tickers))
        row[tickers.index(ticker)] = 1.0
        rows.append(row)
        q.append(view)

    for lhs_group, rhs_group, spread in RELATIVE_VIEWS:
        row = np.zeros(len(tickers))
        n_lhs = sum(1 for t in lhs_group if t in tickers)
        n_rhs = sum(1 for t in rhs_group if t in tickers)
        if n_lhs == 0 or n_rhs == 0:
            continue
        for t in lhs_group:
            if t in tickers:
                row[tickers.index(t)] = 1.0 / n_lhs
        for t in rhs_group:
            if t in tickers:
                row[tickers.index(t)] = -1.0 / n_rhs
        rows.append(row)
        q.append(spread)

    if not rows:
        return np.empty((0, len(tickers))), np.empty((0,))
    return np.vstack(rows), np.array(q)


def posterior(
    returns: pd.DataFrame,
    *,
    tau: float = 0.05,
    confidence: float = VIEW_CONFIDENCE,
    periods: int = 252,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Return (mu_bl, cov_bl) — both annualized — for the configured views."""
    tickers = list(returns.columns)
    Sigma_d = returns.cov().values
    Sigma = Sigma_d * periods

    pi = (returns.mean().values * periods)  # historical mean as equilibrium proxy

    P, Q = build_views(tickers)

    if P.shape[0] == 0:
        mu_bl = pd.Series(pi, index=tickers)
        cov_bl = pd.DataFrame(Sigma, index=tickers, columns=tickers)
        return mu_bl, cov_bl

    # Omega: diagonal, each entry = (1 - confidence)/confidence * tau * P_i Sigma P_i^T
    # Higher confidence -> smaller Omega -> view dominates prior.
    confidence = float(np.clip(confidence, 1e-3, 1 - 1e-3))
    omega_scale = (1.0 - confidence) / confidence
    diag = np.array([omega_scale * tau * (p @ Sigma @ p) for p in P])
    Omega = np.diag(np.maximum(diag, 1e-8))

    tau_sigma_inv = np.linalg.inv(tau * Sigma)
    middle = np.linalg.inv(tau_sigma_inv + P.T @ np.linalg.inv(Omega) @ P)
    mu_bl_vec = middle @ (tau_sigma_inv @ pi + P.T @ np.linalg.inv(Omega) @ Q)
    cov_bl_mat = Sigma + middle

    mu_bl = pd.Series(mu_bl_vec, index=tickers, name="mu_bl")
    cov_bl = pd.DataFrame(cov_bl_mat, index=tickers, columns=tickers)
    return mu_bl, cov_bl
