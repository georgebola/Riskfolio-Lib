"""
Portfolio Analysis Runner
=========================
EDIT ONLY the "PORTFOLIO CONFIG" section.
Everything below is generic and runs unchanged for any set of tickers.

HOW TO RUN
  Colab (option A — paste):
    Copy this entire file into a Colab cell and press Shift+Enter.

  Colab (option B — one-liner):
    import urllib.request
    url = "https://raw.githubusercontent.com/georgebola/Riskfolio-Lib/claude/portfolio-strategy-Wc7Cj/aggressive_portfolio/colab_run.py"
    exec(urllib.request.urlopen(url).read())

  Local:
    python aggressive_portfolio/colab_run.py
"""

# ════════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO CONFIG  ← only edit this section
# ════════════════════════════════════════════════════════════════════════════════

CAPITAL  = 7_500   # total dollars to invest
RF       = 0.04    # annual risk-free rate
LOOKBACK = 3       # years of price history to fetch

# Your portfolio weights (must sum to ~1.0).
# This is the "Baseline" column in the output — your human thesis portfolio.
BASELINE = {
    "BOTZ": 0.176,
    "GRID": 0.157,
    "CIBR": 0.127,
    "RKLB": 0.118,
    "SGOV": 0.098,
    "XAR":  0.088,
    "URA":  0.078,
    "SHLD": 0.078,
    "QTUM": 0.039,
    "XBI":  0.039,
}

# Theme groupings for exposure analysis.
# Set to {} to skip the ThemeExposure sheet.
THEMES = {
    "Robotics":      ["BOTZ"],
    "Grid":          ["GRID"],
    "Cybersecurity": ["CIBR"],
    "Aerospace":     ["XAR", "RKLB"],
    "Nuclear":       ["URA"],
    "Defense":       ["SHLD"],
    "Quantum":       ["QTUM"],
    "Biotech":       ["XBI"],
    "FixedIncome":   ["SGOV"],
}

# Black-Litterman absolute views: ticker -> expected annual return.
# Set to {} to skip all BL methods.
BL_VIEWS = {
    "BOTZ": 0.16,
    "URA":  0.15,
    "CIBR": 0.12,
    "SHLD": 0.14,
    "XAR":  0.13,
    "RKLB": 0.20,
    "GRID": 0.13,
    "QTUM": 0.08,
    "XBI":  0.12,
}

# Black-Litterman relative views: (long_tickers, short_tickers, spread).
# Example below: Aerospace expected to beat Defense by 3 percentage points.
# Set to [] to skip relative views.
BL_RELATIVE = [
    (["XAR", "RKLB"], ["SHLD"], 0.03),
]

# 0 = trust only historical prior, 1 = trust only your views.
BL_CONFIDENCE = 0.85

# Per-asset weight bounds applied to all optimizer runs.
WEIGHT_MIN = 0.00
WEIGHT_MAX = 0.22

# Group-level weight bounds: (tickers_in_group, min_total, max_total).
# Set to [] to disable group constraints.
GROUP_BOUNDS = [
    (["BOTZ"],        0.15, 0.22),
    (["CIBR"],        0.10, 0.22),
    (["SGOV"],        0.08, 0.20),
    (["QTUM"],        0.00, 0.05),
    (["XBI"],         0.03, 0.08),
    (["SHLD"],        0.03, 0.10),
]

# Relative group constraints: group_A total >= group_B total + gap.
# Set to [] to disable.
GROUP_RELATIVE = [
    (["XAR", "RKLB"], ["SHLD"], 0.03),   # Aerospace >= Defense + 3%
]

OUTPUT_FILE = "portfolio_analysis.xlsx"

# ════════════════════════════════════════════════════════════════════════════════
#  ENGINE  ← do not edit below this line
# ════════════════════════════════════════════════════════════════════════════════

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "riskfolio-lib", "yfinance", "openpyxl", "pyportfolioopt"])

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
import riskfolio as rp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

TICKERS = list(BASELINE.keys())

# ── 1. Fetch prices ───────────────────────────────────────────────────────────
end   = datetime.today()
start = end - timedelta(days=int(365.25 * LOOKBACK) + 5)

print("Fetching prices from yfinance…")
raw     = yf.download(TICKERS, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=True)
prices  = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw)[TICKERS].dropna()
returns = prices.pct_change().dropna()
print(f"  {returns.shape[0]} trading days × {returns.shape[1]} tickers\n")

# ── 2. Portfolio statistics ───────────────────────────────────────────────────
def port_stats(weights, rf=RF):
    """Full risk metrics for a weight vector against the global returns."""
    w       = pd.Series(weights).reindex(returns.columns).fillna(0)
    r       = returns @ w
    ann_ret = (1 + r.mean()) ** 252 - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")

    down     = r[r < 0]
    down_vol = down.std() * np.sqrt(252) if len(down) > 1 else float("nan")
    sortino  = (ann_ret - rf) / down_vol if down_vol > 0 else float("nan")

    cum    = (1 + r).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else float("nan")

    var95  = np.percentile(r, 5)
    cvar95 = r[r <= var95].mean()

    return {
        "AnnReturn":   round(ann_ret, 4),
        "AnnVol":      round(ann_vol, 4),
        "Sharpe":      round(sharpe,  4),
        "Sortino":     round(sortino, 4),
        "MaxDrawdown": round(max_dd,  4),
        "Calmar":      round(calmar,  4),
        "DailyVaR95":  round(var95,   4),
        "DailyCVaR95": round(cvar95,  4),
    }

# ── 3. Black-Litterman posterior (numpy) ──────────────────────────────────────
def _bl_posterior(abs_views, rel_views, confidence, tau=0.05, periods=252):
    tickers = list(returns.columns)
    Sigma   = returns.cov().values * periods
    pi      = returns.mean().values * periods

    rows, q = [], []
    for t, v in abs_views.items():
        if t in tickers:
            row = np.zeros(len(tickers))
            row[tickers.index(t)] = 1.0
            rows.append(row); q.append(v)
    for longs, shorts, spread in rel_views:
        row = np.zeros(len(tickers))
        for t in longs:
            if t in tickers: row[tickers.index(t)] =  1 / len(longs)
        for t in shorts:
            if t in tickers: row[tickers.index(t)] = -1 / len(shorts)
        rows.append(row); q.append(spread)

    if not rows:
        return pd.Series(pi, index=tickers), pd.DataFrame(Sigma, index=tickers, columns=tickers)

    P, Q  = np.vstack(rows), np.array(q)
    c     = np.clip(confidence, 1e-3, 1 - 1e-3)
    omega = np.diag([(1-c)/c * tau * (p @ Sigma @ p) for p in P])
    tsi   = np.linalg.inv(tau * Sigma)
    mid   = np.linalg.inv(tsi + P.T @ np.linalg.inv(omega) @ P)
    mu_bl = mid @ (tsi @ pi + P.T @ np.linalg.inv(omega) @ Q)
    return (pd.Series(mu_bl, index=tickers),
            pd.DataFrame(Sigma + mid, index=tickers, columns=tickers))

# ── 4. Riskfolio helpers ──────────────────────────────────────────────────────
def _rf_constraints(tickers):
    """Build (A, B) inequality matrix where A @ w <= B."""
    rows_A, rows_B = [], []
    for members, lo, hi in GROUP_BOUNDS:
        ind = np.array([1.0 if t in members else 0.0 for t in tickers])
        if ind.sum() == 0: continue
        rows_A.append(-ind); rows_B.append(-lo)   # sum >= lo
        rows_A.append( ind); rows_B.append( hi)   # sum <= hi
    for group_a, group_b, gap in GROUP_RELATIVE:
        a = np.array([1.0 if t in group_a else 0.0 for t in tickers])
        b = np.array([1.0 if t in group_b else 0.0 for t in tickers])
        rows_A.append(b - a); rows_B.append(-gap)  # a - b >= gap
    if not rows_A:
        return None, None
    return np.vstack(rows_A), np.array(rows_B).reshape(-1, 1)

def _make_rf_port(mu_override=None, cov_override=None):
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="ledoit")
    if mu_override is not None:
        port.mu  = mu_override.to_frame().T
    if cov_override is not None:
        port.cov = cov_override
    port.sht      = False
    port.upperlng = WEIGHT_MAX
    A, B = _rf_constraints(list(returns.columns))
    if A is not None:
        port.ainequality = A
        port.binequality = B
    return port

def _rf_max_sharpe(port):
    """Utility sweep workaround for riskfolio obj='Sharpe' bug."""
    best_s, best_w = -np.inf, None
    for l in np.geomspace(0.1, 200, 30):
        w = port.optimization(model="Classic", rm="MV", obj="Utility",
                              rf=RF, l=float(l), hist=True)
        if w is None: continue
        s = port_stats(w["weights"])["Sharpe"]
        if s > best_s:
            best_s, best_w = s, w["weights"]
    return best_w

# ── 5. Run Riskfolio optimizers ───────────────────────────────────────────────
print("Running Riskfolio optimizers…")
rf_results = {}

base = _make_rf_port()
w = _rf_max_sharpe(base)
if w is not None: rf_results["RF_MaxSharpe"] = w

w = base.optimization(model="Classic", rm="MV", obj="MinRisk", rf=RF, hist=True)
if w is not None: rf_results["RF_MinVol"] = w["weights"]

w = base.optimization(model="Classic", rm="CVaR", obj="MinRisk", rf=RF, hist=True)
if w is not None: rf_results["RF_MinCVaR"] = w["weights"]

rp2 = _make_rf_port()
w = rp2.rp_optimization(model="Classic", rm="MV", rf=RF, hist=True)
if w is not None: rf_results["RF_RiskParity"] = w["weights"]

has_bl = bool(BL_VIEWS)
if has_bl:
    print("  Computing BL posterior…")
    mu_bl, cov_bl = _bl_posterior(BL_VIEWS, BL_RELATIVE, BL_CONFIDENCE)
    bl_port = _make_rf_port(mu_override=mu_bl, cov_override=cov_bl)
    w = _rf_max_sharpe(bl_port)
    if w is not None: rf_results["RF_BL_MaxSharpe"] = w

print(f"  Done — {len(rf_results)} methods\n")

# ── 6. Run PyPortfolioOpt optimizers ──────────────────────────────────────────
from pypfopt import EfficientFrontier, risk_models, expected_returns, BlackLittermanModel

print("Running PyPortfolioOpt optimizers…")
ppo_results = {}

def _ppo_ef(mu, S):
    """EfficientFrontier with the same group constraints."""
    ef = EfficientFrontier(mu, S, weight_bounds=(WEIGHT_MIN, WEIGHT_MAX))
    tickers = list(returns.columns)
    for members, lo, hi in GROUP_BOUNDS:
        idx = [tickers.index(t) for t in members if t in tickers]
        if not idx: continue
        ef.add_constraint(lambda w, i=idx, lo=lo: sum(w[j] for j in i) - lo)
        ef.add_constraint(lambda w, i=idx, hi=hi: hi - sum(w[j] for j in i))
    for group_a, group_b, gap in GROUP_RELATIVE:
        ia = [tickers.index(t) for t in group_a if t in tickers]
        ib = [tickers.index(t) for t in group_b if t in tickers]
        ef.add_constraint(lambda w, a=ia, b=ib, g=gap:
                          sum(w[j] for j in a) - sum(w[j] for j in b) - g)
    return ef

def _ppo_weights(raw):
    """Normalize cleaned weights dict → pd.Series aligned to TICKERS."""
    w = pd.Series({t: float(raw.get(t, 0.0)) for t in TICKERS})
    return (w / w.sum()).round(4)

try:
    S_hist  = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    mu_hist = expected_returns.mean_historical_return(prices)

    try:
        ef = _ppo_ef(mu_hist, S_hist)
        ef.max_sharpe(risk_free_rate=RF)
        ppo_results["PPO_MaxSharpe"] = _ppo_weights(ef.clean_weights())
    except Exception as e:
        print(f"  PPO_MaxSharpe failed: {e}")

    try:
        ef = _ppo_ef(mu_hist, S_hist)
        ef.min_volatility()
        ppo_results["PPO_MinVol"] = _ppo_weights(ef.clean_weights())
    except Exception as e:
        print(f"  PPO_MinVol failed: {e}")

    if has_bl:
        try:
            abs_views_u = {k: v for k, v in BL_VIEWS.items() if k in TICKERS}
            confs       = [BL_CONFIDENCE] * len(abs_views_u)
            bl_m        = BlackLittermanModel(S_hist, pi=mu_hist,
                                              absolute_views=abs_views_u,
                                              omega="idzorek",
                                              view_confidences=confs)
            ef = _ppo_ef(bl_m.bl_returns(), bl_m.bl_cov())
            ef.max_sharpe(risk_free_rate=RF)
            ppo_results["PPO_BL_MaxSharpe"] = _ppo_weights(ef.clean_weights())
        except Exception as e:
            print(f"  PPO_BL_MaxSharpe failed: {e}")

except Exception as e:
    print(f"  PyPortfolioOpt setup failed: {e}")

print(f"  Done — {len(ppo_results)} methods\n")

# ── 7. Combine and display ────────────────────────────────────────────────────
all_weights = {"Baseline": pd.Series(BASELINE), **rf_results, **ppo_results}
weights_df  = pd.DataFrame(all_weights).reindex(TICKERS).fillna(0).round(4)
stats_df    = pd.DataFrame({col: port_stats(weights_df[col]) for col in weights_df}).T
dollar_df   = (weights_df * CAPITAL).round(0)

theme_df = pd.DataFrame()
if THEMES:
    theme_df = pd.DataFrame({
        col: {th: float(weights_df[col].reindex(m).fillna(0).sum())
              for th, m in THEMES.items()}
        for col in weights_df
    }).T.round(4)

print("=== WEIGHTS ===")
print(weights_df.to_string())
print("\n=== STATS ===")
print(stats_df.to_string())
if not theme_df.empty:
    print("\n=== THEME EXPOSURE ===")
    print(theme_df.to_string())

# ── 8. Write Excel ────────────────────────────────────────────────────────────
RISK_COLS = ["AnnReturn","AnnVol","Sharpe","Sortino","MaxDrawdown","Calmar","DailyCVaR95"]
out = Path(OUTPUT_FILE)
with pd.ExcelWriter(out, engine="openpyxl") as xw:
    weights_df.to_excel(xw, sheet_name="Weights")
    dollar_df.to_excel(xw,  sheet_name="Dollars")
    stats_df[RISK_COLS].to_excel(xw, sheet_name="RiskCompare")
    stats_df.to_excel(xw,   sheet_name="Stats_Full")
    if not theme_df.empty:
        theme_df.to_excel(xw, sheet_name="ThemeExposure")
    if has_bl:
        mu_bl.to_frame("mu_BL").assign(mu_hist=returns.mean() * 252).round(4).to_excel(
            xw, sheet_name="BL_Returns")
print(f"\nExcel written → {out.resolve()}")

# ── 9. Frontier plot ──────────────────────────────────────────────────────────
COLORS = [
    "#d62728","#1f77b4","#2ca02c","#9467bd",
    "#ff7f0e","#17becf","#e377c2","#8c564b","#bcbd22","#7f7f7f",
]
fig, ax = plt.subplots(figsize=(11, 7))
try:
    fport    = _make_rf_port()
    frontier = fport.efficient_frontier(model="Classic", rm="MV",
                                        points=30, rf=RF, hist=True)
    fs = [port_stats(frontier[c]) for c in frontier.columns]
    ax.plot([s["AnnVol"] for s in fs], [s["AnnReturn"] for s in fs],
            "-", color="#ccc", lw=2, label="MV Frontier", zorder=1)
except Exception:
    pass

for col, color in zip(weights_df.columns, COLORS):
    s = port_stats(weights_df[col])
    ax.scatter(s["AnnVol"], s["AnnReturn"], s=140, color=color,
               edgecolor="black", zorder=5, label=col)

ax.set_xlabel("Annualised Volatility")
ax.set_ylabel("Annualised Return")
ax.set_title("Portfolio Comparison — Real Market Data")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("efficient_frontier.png", dpi=140)
print("Frontier plot → efficient_frontier.png")

# ── 10. Download (Colab) ──────────────────────────────────────────────────────
try:
    from google.colab import files
    files.download(str(out))
    files.download("efficient_frontier.png")
    print("Downloads triggered.")
except ImportError:
    print("Not in Colab — files saved in current directory.")
