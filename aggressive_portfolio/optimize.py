"""Run all four optimization methods plus Black-Litterman against the same
constraint set. Returns a dict of method-name -> weight Series.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

import riskfolio as rp

from .config import RISK_FREE_RATE, Constraints
from .constraints import build_constraints

logger = logging.getLogger(__name__)


def _apply_constraints(port: rp.Portfolio, returns: pd.DataFrame, cfg: Constraints) -> None:
    A, B, (lb, ub) = build_constraints(list(returns.columns), cfg)
    port.ainequality = A.values if len(A) else None
    port.binequality = B.values if len(B) else None
    port.upperlng = ub
    # Riskfolio uses a single scalar lower bound for long-only via `sht=False`;
    # per-asset minimums are enforced through additional inequality rows.
    if lb > 0:
        # per-asset min: w_i >= lb  =>  -I @ w <= -lb
        n = returns.shape[1]
        neg_eye = -np.eye(n)
        lb_vec = np.full((n, 1), -lb)
        if port.ainequality is None:
            port.ainequality = neg_eye
            port.binequality = lb_vec
        else:
            port.ainequality = np.vstack([port.ainequality, neg_eye])
            port.binequality = np.vstack([port.binequality, lb_vec])


def _make_portfolio(
    returns: pd.DataFrame,
    cfg: Constraints,
    mu_override: Optional[pd.Series] = None,
    cov_override: Optional[pd.DataFrame] = None,
) -> rp.Portfolio:
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="ledoit")
    if mu_override is not None:
        port.mu = mu_override.to_frame().T  # Riskfolio expects 1-row frame
    if cov_override is not None:
        port.cov = cov_override
    port.sht = False  # long-only
    port.uppersht = 0.0
    _apply_constraints(port, returns, cfg)
    return port


def _opt(port: rp.Portfolio, *, model: str, rm: str, obj: str) -> pd.Series:
    w = port.optimization(model=model, rm=rm, obj=obj, rf=RISK_FREE_RATE, hist=True)
    if w is None:
        raise RuntimeError(f"Optimization failed: model={model} rm={rm} obj={obj}")
    return w["weights"]


def _max_sharpe_via_utility_sweep(
    port: rp.Portfolio,
    returns: pd.DataFrame,
    *,
    rm: str = "MV",
    rf: float = RISK_FREE_RATE,
    lambdas: Optional[np.ndarray] = None,
) -> pd.Series:
    """Workaround for the obj='Sharpe' Charnes-Cooper failure on riskfolio 7.2.1.

    Sweep the risk-aversion coefficient l in obj='Utility' and pick the weights
    that maximize the empirical Sharpe ratio.
    """
    if lambdas is None:
        lambdas = np.geomspace(0.1, 200, 25)
    best_sharpe = -np.inf
    best_w: Optional[pd.Series] = None
    for l in lambdas:
        w = port.optimization(model="Classic", rm=rm, obj="Utility",
                              rf=rf, l=float(l), hist=True)
        if w is None:
            continue
        weights = w["weights"]
        s = portfolio_stats(weights, returns, rf=rf)["Sharpe"]
        if np.isfinite(s) and s > best_sharpe:
            best_sharpe = s
            best_w = weights
    if best_w is None:
        raise RuntimeError("Utility sweep produced no feasible portfolio")
    return best_w


def run_all(
    returns: pd.DataFrame,
    cfg: Constraints,
    mu_bl: Optional[pd.Series] = None,
    cov_bl: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.Series]:
    """Run Max-Sharpe, Min-Volatility, Min-CVaR, Risk-Parity, plus BL Max-Sharpe.

    Each result is a pd.Series of weights indexed by ticker.
    """
    results: Dict[str, pd.Series] = {}

    base = _make_portfolio(returns, cfg)

    results["MaxSharpe_MV"] = _max_sharpe_via_utility_sweep(base, returns)
    results["MinVol_MV"]    = _opt(base, model="Classic", rm="MV", obj="MinRisk")
    results["MinCVaR"]      = _opt(base, model="Classic", rm="CVaR", obj="MinRisk")

    # Risk parity uses a different entry point
    rp_port = _make_portfolio(returns, cfg)
    w_rp = rp_port.rp_optimization(model="Classic", rm="MV", rf=RISK_FREE_RATE, hist=True)
    if w_rp is None:
        raise RuntimeError("Risk parity optimization failed")
    results["RiskParity"] = w_rp["weights"]

    if mu_bl is not None and cov_bl is not None:
        bl_port = _make_portfolio(returns, cfg, mu_override=mu_bl, cov_override=cov_bl)
        results["BlackLitterman_MaxSharpe"] = _max_sharpe_via_utility_sweep(
            bl_port, returns
        )

    return results


def portfolio_stats(
    weights: pd.Series,
    returns: pd.DataFrame,
    rf: float = RISK_FREE_RATE,
    periods: int = 252,
) -> Dict[str, float]:
    """Annualized return, volatility, Sharpe, plus historical CVaR(95%)."""
    aligned = weights.reindex(returns.columns).fillna(0.0)
    port_ret_daily = returns @ aligned
    mean_d = port_ret_daily.mean()
    vol_d = port_ret_daily.std()
    ann_ret = (1 + mean_d) ** periods - 1
    ann_vol = vol_d * np.sqrt(periods)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")
    var95 = np.percentile(port_ret_daily, 5)
    cvar95 = port_ret_daily[port_ret_daily <= var95].mean()
    return {
        "AnnReturn": float(ann_ret),
        "AnnVol": float(ann_vol),
        "Sharpe": float(sharpe),
        "DailyVaR95": float(var95),
        "DailyCVaR95": float(cvar95),
    }
