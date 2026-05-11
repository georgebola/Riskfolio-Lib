"""
Paste this entire file into a Google Colab cell and run it.
It installs deps, fetches real prices, runs all optimizers,
and produces a downloadable Excel + frontier plot.

Steps:
  1. Go to https://colab.research.google.com
  2. File > New notebook
  3. Paste this entire script into the first cell
  4. Runtime > Run all  (or Shift+Enter)
  5. Download the Excel from the Files panel on the left
"""

# ── 1. Install ────────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "riskfolio-lib", "yfinance", "openpyxl"])

# ── 2. Imports ────────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
import riskfolio as rp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# ── 3. Config ─────────────────────────────────────────────────────────────────
RF          = 0.04          # risk-free rate
LOOKBACK    = 3             # years of history
CAPITAL     = 7_500         # dollars

# v12 weights — the original baseline
V12 = {
    "BOTZ": 0.20, "URA":  0.12, "CIBR": 0.15,
    "SHLD": 0.05, "XAR":  0.15, "RKLB": 0.05,
    "GRID": 0.08, "QTUM": 0.03, "XBI":  0.05,
    "SGOV": 0.12,
}

# v13 weights — optimizer-guided revision with higher GRID/RKLB
V13 = {
    "BOTZ": 0.18, "URA":  0.08, "CIBR": 0.13,
    "SHLD": 0.08, "XAR":  0.09, "RKLB": 0.12,
    "GRID": 0.16, "QTUM": 0.04, "XBI":  0.04,
    "SGOV": 0.10,
}

# vFinal — whole-share normalized weights (your actual deployment)
V_FINAL = {
    "BOTZ": 0.176, "GRID": 0.157, "CIBR": 0.127,
    "RKLB": 0.118, "SGOV": 0.098, "XAR":  0.088,
    "URA":  0.078, "SHLD": 0.078, "QTUM": 0.039,
    "XBI":  0.039,
}

THEMES = {
    "Robotics":     ["BOTZ"],
    "Aerospace":    ["XAR", "RKLB"],
    "Defense":      ["SHLD"],
    "Nuclear":      ["URA"],
    "Grid":         ["GRID"],
    "Quantum":      ["QTUM"],
    "Cybersecurity":["CIBR"],
    "Biotech":      ["XBI"],
    "FixedIncome":  ["SGOV"],
}

TICKERS = list(V12.keys())

# Black-Litterman forward views
BL_VIEWS = {
    "QTUM": 0.08,   # cap the 86% trailing return
    "XBI":  0.12,
    "BOTZ": 0.16,
    "URA":  0.15,
    "GRID": 0.13,
    "CIBR": 0.12,
    "SHLD": 0.14,
    "XAR":  0.13,
    "RKLB": 0.20,
}
BL_CONFIDENCE = 0.85

# ── 4. Fetch prices ───────────────────────────────────────────────────────────
end   = datetime.today()
start = end - timedelta(days=int(365.25 * LOOKBACK) + 5)

print("Fetching prices from yfinance…")
raw = yf.download(TICKERS, start=start.strftime("%Y-%m-%d"),
                  end=end.strftime("%Y-%m-%d"),
                  auto_adjust=True, progress=True)

prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
prices = prices[TICKERS].dropna()
returns = prices.pct_change().dropna()
print(f"Returns: {returns.shape[0]} trading days × {returns.shape[1]} tickers\n")

# ── 5. Black-Litterman posterior ──────────────────────────────────────────────
def bl_posterior(returns, views, confidence=0.85, tau=0.05, periods=252):
    tickers = list(returns.columns)
    Sigma   = returns.cov().values * periods
    pi      = returns.mean().values * periods

    rows, q = [], []
    for t, v in views.items():
        if t in tickers:
            r = np.zeros(len(tickers))
            r[tickers.index(t)] = 1.0
            rows.append(r); q.append(v)
    # Aerospace > Defense by 3%
    aero = ["XAR", "RKLB"]; defe = ["SHLD"]
    row  = np.zeros(len(tickers))
    for t in aero:
        if t in tickers: row[tickers.index(t)] =  1 / len(aero)
    for t in defe:
        if t in tickers: row[tickers.index(t)] = -1 / len(defe)
    rows.append(row); q.append(0.03)

    P, Q   = np.vstack(rows), np.array(q)
    c      = np.clip(confidence, 1e-3, 1-1e-3)
    omega  = np.diag([(1-c)/c * tau * (p @ Sigma @ p) for p in P])

    tsi    = np.linalg.inv(tau * Sigma)
    mid    = np.linalg.inv(tsi + P.T @ np.linalg.inv(omega) @ P)
    mu_bl  = mid @ (tsi @ pi + P.T @ np.linalg.inv(omega) @ Q)
    cov_bl = Sigma + mid
    return (pd.Series(mu_bl, index=tickers),
            pd.DataFrame(cov_bl, index=tickers, columns=tickers))

mu_bl, cov_bl = bl_posterior(returns, BL_VIEWS, BL_CONFIDENCE)
print("Black-Litterman mu (annual):")
print(mu_bl.round(3).to_string(), "\n")

# ── 6. Constraint builder ─────────────────────────────────────────────────────
def build_constraints(tickers):
    """Returns (A, B) where A @ w <= B, and scalar (lb, ub)."""
    rows_A, rows_B = [], []

    group_bounds = [
        (["BOTZ"],          0.18, 0.22),
        (["CIBR"],          0.12, 0.20),
        (["SGOV"],          0.10, 0.18),
        (["QTUM"],          0.00, 0.05),
        (["XBI"],           0.03, 0.08),
        (["SHLD"],          0.03, 0.08),
    ]
    for members, lo, hi in group_bounds:
        ind = np.array([1.0 if t in members else 0.0 for t in tickers])
        if ind.sum() == 0: continue
        rows_A.append(-ind);  rows_B.append(-lo)  # sum >= lo
        rows_A.append( ind);  rows_B.append( hi)  # sum <= hi

    # Aerospace - Defense >= 0.03  =>  Defense - Aerospace <= -0.03
    aero = np.array([1.0 if t in ["XAR","RKLB"] else 0.0 for t in tickers])
    defe = np.array([1.0 if t in ["SHLD"]       else 0.0 for t in tickers])
    rows_A.append(defe - aero); rows_B.append(-0.03)

    A = pd.DataFrame(np.vstack(rows_A), columns=tickers)
    B = pd.DataFrame(np.array(rows_B).reshape(-1, 1))
    return A, B

# ── 7. Optimize ───────────────────────────────────────────────────────────────
def make_port(returns, mu_override=None, cov_override=None):
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="ledoit")
    if mu_override is not None:
        port.mu  = mu_override.to_frame().T
    if cov_override is not None:
        port.cov = cov_override
    port.sht     = False
    port.upperlng = 0.22
    A, B = build_constraints(list(returns.columns))
    port.ainequality = A.values
    port.binequality = B.values
    return port

def max_sharpe_sweep(port, returns, rf=RF):
    best_s, best_w = -np.inf, None
    for l in np.geomspace(0.1, 200, 30):
        w = port.optimization(model="Classic", rm="MV", obj="Utility",
                              rf=rf, l=float(l), hist=True)
        if w is None: continue
        weights  = w["weights"]
        port_ret = (returns @ weights.reindex(returns.columns).fillna(0)).mean()
        port_vol = (returns @ weights.reindex(returns.columns).fillna(0)).std()
        ann_ret  = (1 + port_ret)**252 - 1
        ann_vol  = port_vol * np.sqrt(252)
        sharpe   = (ann_ret - rf) / ann_vol if ann_vol > 0 else -np.inf
        if sharpe > best_s:
            best_s, best_w = sharpe, weights
    return best_w

def port_stats(weights, returns, rf=RF):
    w         = weights.reindex(returns.columns).fillna(0)
    r         = returns @ w
    ann_ret   = (1 + r.mean())**252 - 1
    ann_vol   = r.std() * np.sqrt(252)
    sharpe    = (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")

    # Sortino: only penalise negative daily returns
    downside  = r[r < 0]
    down_vol  = downside.std() * np.sqrt(252) if len(downside) > 1 else float("nan")
    sortino   = (ann_ret - rf) / down_vol if down_vol > 0 else float("nan")

    # Max drawdown and Calmar
    cum       = (1 + r).cumprod()
    peak      = cum.cummax()
    dd        = (cum - peak) / peak
    max_dd    = float(dd.min())
    calmar    = ann_ret / abs(max_dd) if max_dd < 0 else float("nan")

    # Tail risk
    var95     = np.percentile(r, 5)
    cvar95    = r[r <= var95].mean()

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

print("Running optimizers (this takes ~30 seconds)…")
base = make_port(returns)
results = {}

results["MaxSharpe_MV"]  = max_sharpe_sweep(base, returns)
results["MinVol_MV"]     = base.optimization(model="Classic", rm="MV",
                            obj="MinRisk", rf=RF, hist=True)["weights"]
results["MinCVaR"]       = base.optimization(model="Classic", rm="CVaR",
                            obj="MinRisk", rf=RF, hist=True)["weights"]
rp_port = make_port(returns)
results["RiskParity"]    = rp_port.rp_optimization(model="Classic", rm="MV",
                            rf=RF, hist=True)["weights"]
bl_port = make_port(returns, mu_override=mu_bl, cov_override=cov_bl)
results["BL_MaxSharpe"]  = max_sharpe_sweep(bl_port, returns)

print("Done.\n")

# ── 8. Build output tables ────────────────────────────────────────────────────
weights_df = pd.DataFrame(results)
weights_df.insert(0, "v12",     pd.Series(V12))
weights_df.insert(1, "v13",     pd.Series(V13))
weights_df.insert(2, "vFinal",  pd.Series(V_FINAL))
weights_df = weights_df.fillna(0).round(4)

stats_rows = {}
for col in weights_df.columns:
    stats_rows[col] = port_stats(weights_df[col], returns)
stats_df = pd.DataFrame(stats_rows).T

theme_rows = {}
for col in weights_df.columns:
    theme_rows[col] = {
        th: float(weights_df[col].reindex(members).fillna(0).sum())
        for th, members in THEMES.items()
    }
theme_df = pd.DataFrame(theme_rows).T.round(4)

dollar_df = (weights_df * CAPITAL).round(0)

print("=== WEIGHTS ===")
print(weights_df.to_string())
print("\n=== STATS ===")
print(stats_df.to_string())
print("\n=== THEME EXPOSURE ===")
print(theme_df.to_string())

# ── 9. Write Excel ────────────────────────────────────────────────────────────
out = Path("portfolio_v13_real.xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as xw:
    weights_df.to_excel(xw, sheet_name="Weights")
    dollar_df.to_excel(xw,  sheet_name="Dollars ($7500)")
    stats_df.to_excel(xw,   sheet_name="Stats")
    theme_df.to_excel(xw,   sheet_name="ThemeExposure")
    mu_bl.to_frame("mu_BL").assign(mu_hist=returns.mean()*252).to_excel(
        xw, sheet_name="ExpectedReturns")

    # Highlight: Sharpe vs Sortino side-by-side for the three human portfolios
    compare_cols = ["v12", "v13", "vFinal"]
    compare_metrics = ["AnnReturn","AnnVol","Sharpe","Sortino","MaxDrawdown","Calmar","DailyCVaR95"]
    stats_df[compare_metrics].loc[compare_cols].to_excel(xw, sheet_name="Risk_Compare")
print(f"\nExcel written → {out.resolve()}")

# ── 10. Frontier plot ─────────────────────────────────────────────────────────
port_f = make_port(returns)
try:
    frontier = port_f.efficient_frontier(model="Classic", rm="MV",
                                          points=25, rf=RF, hist=True)
    f_stats  = [port_stats(frontier[c], returns) for c in frontier.columns]
    fig, ax  = plt.subplots(figsize=(9, 6))
    ax.plot([s["AnnVol"]    for s in f_stats],
            [s["AnnReturn"] for s in f_stats],
            "-", color="#aaa", lw=2, label="Efficient frontier")
except Exception:
    fig, ax = plt.subplots(figsize=(9, 6))

colors = {
    "v12":"#d62728","v13":"#e377c2","vFinal":"#8c564b",
    "MaxSharpe_MV":"#1f77b4","MinVol_MV":"#2ca02c",
    "MinCVaR":"#9467bd","RiskParity":"#ff7f0e","BL_MaxSharpe":"#17becf",
}
for col in weights_df.columns:
    s = port_stats(weights_df[col], returns)
    ax.scatter(s["AnnVol"], s["AnnReturn"], s=120,
               color=colors.get(col,"black"), edgecolor="black",
               zorder=5, label=col)

ax.set_xlabel("Annualised volatility"); ax.set_ylabel("Annualised return")
ax.set_title("v12 / v13 / vFinal vs optimisers — real data")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig("efficient_frontier_real.png", dpi=140)
print("Frontier plot → efficient_frontier_real.png")

# ── 11. Colab download helper ─────────────────────────────────────────────────
try:
    from google.colab import files
    files.download("portfolio_v13_real.xlsx")
    files.download("efficient_frontier_real.png")
    print("Download triggered.")
except ImportError:
    print("Not in Colab — files are in the current directory.")
