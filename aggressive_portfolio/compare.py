"""Generate the Excel comparison + efficient frontier plot."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import RISK_FREE_RATE, THEMES, V4_WEIGHTS
from .optimize import portfolio_stats

logger = logging.getLogger(__name__)


def weights_table(results: Dict[str, pd.Series], v4: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df.insert(0, "v4", pd.Series(v4))
    df = df.fillna(0.0)
    df["v4_minus_optMaxSharpe"] = df["v4"] - df.get(
        "MaxSharpe_MV", pd.Series(0.0, index=df.index)
    )
    return df.round(4)


def stats_table(
    weights_df: pd.DataFrame, returns: pd.DataFrame, rf: float = RISK_FREE_RATE
) -> pd.DataFrame:
    rows = {}
    for col in weights_df.columns:
        if col.startswith("v4_minus"):
            continue
        rows[col] = portfolio_stats(weights_df[col], returns, rf=rf)
    return pd.DataFrame(rows).T.round(4)


def theme_exposure(weights_df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col in weights_df.columns:
        if col.startswith("v4_minus"):
            continue
        out[col] = {
            theme: float(weights_df[col].reindex(members).fillna(0.0).sum())
            for theme, members in THEMES.items()
        }
    return pd.DataFrame(out).round(4)


def write_excel(
    out_path: Path,
    weights_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    themes_df: pd.DataFrame,
    mu_hist: pd.Series,
    mu_bl: pd.Series,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        weights_df.to_excel(xw, sheet_name="Weights")
        stats_df.to_excel(xw, sheet_name="Stats")
        themes_df.to_excel(xw, sheet_name="ThemeExposure")
        pd.DataFrame({"mu_historical": mu_hist, "mu_blackLitterman": mu_bl}).to_excel(
            xw, sheet_name="ExpectedReturns"
        )
    logger.info("Wrote %s", out_path)


def plot_efficient_frontier(
    returns: pd.DataFrame,
    weights_df: pd.DataFrame,
    out_path: Path,
    rf: float = RISK_FREE_RATE,
) -> None:
    """Sample the long-only efficient frontier and overlay each portfolio."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import riskfolio as rp

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="ledoit")
    port.sht = False
    frontier = port.efficient_frontier(model="Classic", rm="MV", points=30, rf=rf, hist=True)

    mus = []
    sigmas = []
    for col in frontier.columns:
        w = frontier[col]
        s = portfolio_stats(w, returns, rf=rf)
        mus.append(s["AnnReturn"])
        sigmas.append(s["AnnVol"])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sigmas, mus, "-", color="#666", lw=2, label="Efficient frontier (long-only)")

    colors = {
        "v4": "#d62728",
        "MaxSharpe_MV": "#1f77b4",
        "MinVol_MV": "#2ca02c",
        "MinCVaR": "#9467bd",
        "RiskParity": "#ff7f0e",
        "BlackLitterman_MaxSharpe": "#17becf",
    }
    for col in weights_df.columns:
        if col.startswith("v4_minus"):
            continue
        s = portfolio_stats(weights_df[col], returns, rf=rf)
        ax.scatter(s["AnnVol"], s["AnnReturn"], s=110,
                   color=colors.get(col, "black"), edgecolor="black",
                   zorder=5, label=col)

    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.set_title("Aggressive portfolio vs. efficient frontier")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
