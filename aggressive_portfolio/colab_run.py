"""
Portfolio Analysis Runner — two ways to use
============================================

── V1  Pass everything upfront (no prompts) ──────────────────────────────────
Define a CONFIG dict in the same cell before the exec() call:

    import urllib.request

    CONFIG = {
        "capital"        : 7500,
        "rf"             : 0.04,
        "lookback"       : 3,
        "target_return"  : 0.12,          # optional: adds PPO_TargetReturn method
        "baseline"       : {"BOTZ": 0.176, "GRID": 0.157, "CIBR": 0.127,
                            "RKLB": 0.118, "SGOV": 0.098, "XAR":  0.088,
                            "URA":  0.078, "SHLD": 0.078, "QTUM": 0.039,
                            "XBI":  0.039},
        "themes"         : {"Robotics": ["BOTZ"], "Grid": ["GRID"],
                            "Cybersecurity": ["CIBR"], "Aerospace": ["XAR","RKLB"],
                            "Nuclear": ["URA"], "Defense": ["SHLD"],
                            "Quantum": ["QTUM"], "Biotech": ["XBI"],
                            "FixedIncome": ["SGOV"]},
        "bl_views"       : {"BOTZ":0.16,"URA":0.15,"CIBR":0.12,"SHLD":0.14,
                            "XAR":0.13,"RKLB":0.20,"GRID":0.13,"QTUM":0.08,"XBI":0.12},
        "bl_relative"    : [(["XAR","RKLB"], ["SHLD"], 0.03)],
        "bl_confidence"  : 0.85,
        "weight_min"     : 0.00,
        "weight_max"     : 0.22,
        "group_bounds"   : [(["BOTZ"],0.15,0.22),(["CIBR"],0.10,0.22),
                            (["SGOV"],0.08,0.20),(["QTUM"],0.00,0.05),
                            (["XBI"],0.03,0.08),(["SHLD"],0.03,0.10)],
        "group_relative" : [(["XAR","RKLB"], ["SHLD"], 0.03)],
        "output_file"    : "portfolio_analysis.xlsx",
    }

    url = "https://raw.githubusercontent.com/georgebola/Riskfolio-Lib/claude/portfolio-strategy-Wc7Cj/aggressive_portfolio/colab_run.py"
    exec(urllib.request.urlopen(url).read())

── V2  Interactive prompts (answer questions as it runs) ─────────────────────
Just run the exec() without defining CONFIG first:

    import urllib.request
    url = "https://raw.githubusercontent.com/georgebola/Riskfolio-Lib/claude/portfolio-strategy-Wc7Cj/aggressive_portfolio/colab_run.py"
    exec(urllib.request.urlopen(url).read())

── Local ─────────────────────────────────────────────────────────────────────
    python aggressive_portfolio/colab_run.py

── Methods run ───────────────────────────────────────────────────────────────
  Riskfolio-Lib  : MaxSharpe, MinVol, MinCVaR, MinCDaR, RiskParity, HRP, HERC, NCO, BL_MaxSharpe
  PyPortfolioOpt : MaxSharpe, MinVol, MinCVaR, MinSemivar, HRP, CAPM_MaxSharpe, EMA_MaxSharpe,
                   BL_MaxSharpe, TargetReturn (if target_return set)
"""

# ── Install ───────────────────────────────────────────────────────────────────
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

# ── Input helpers (V2 only) ───────────────────────────────────────────────────
def _ask(prompt, default=None):
    suffix = f" [Enter = {default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else str(default) if default is not None else ""

def _parse_pairs(s):
    out = {}
    for part in s.split(","):
        part = part.strip()
        if ":" not in part: continue
        k, v = part.split(":", 1)
        k = k.strip().upper()
        if k: out[k] = float(v.strip())
    return out

def _parse_relative(s):
    result = []
    for part in s.split(";"):
        if ">" not in part or ":" not in part: continue
        lhs, rest = part.split(">", 1)
        rhs, val  = rest.rsplit(":", 1)
        result.append((
            [t.strip().upper() for t in lhs.split("+") if t.strip()],
            [t.strip().upper() for t in rhs.split("+") if t.strip()],
            float(val),
        ))
    return result

# ── V1: CONFIG dict supplied → skip prompts ───────────────────────────────────
try:
    _cfg = CONFIG  # noqa: F821
    CAPITAL        = float(_cfg.get("capital",        7500))
    RF             = float(_cfg.get("rf",             0.04))
    LOOKBACK       = int(_cfg.get("lookback",         3))
    TARGET_RETURN  = float(_cfg["target_return"]) if _cfg.get("target_return") else None
    BASELINE       = _cfg["baseline"]
    THEMES         = _cfg.get("themes",         {})
    BL_VIEWS       = _cfg.get("bl_views",       {})
    BL_RELATIVE    = _cfg.get("bl_relative",    [])
    BL_CONFIDENCE  = float(_cfg.get("bl_confidence", 0.85))
    WEIGHT_MIN     = float(_cfg.get("weight_min",    0.00))
    WEIGHT_MAX     = float(_cfg.get("weight_max",    0.22))
    GROUP_BOUNDS   = _cfg.get("group_bounds",   [])
    GROUP_RELATIVE = _cfg.get("group_relative", [])
    OUTPUT_FILE    = _cfg.get("output_file",    "portfolio_analysis.xlsx")

    total = sum(BASELINE.values())
    if total > 1.5: BASELINE = {k: v/100 for k, v in BASELINE.items()}; total /= 100
    if abs(total - 1.0) > 0.005: BASELINE = {k: v/total for k, v in BASELINE.items()}
    TICKERS = list(BASELINE.keys())
    print("\n[V1] CONFIG loaded — skipping prompts.")

# ── V2: No CONFIG → ask prompts ───────────────────────────────────────────────
except NameError:
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║       PORTFOLIO ANALYSER — answer the prompts       ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n── Settings ─────────────────────────────────────────────")
    CAPITAL  = float(_ask("Total capital ($)", 7500))
    RF       = float(_ask("Risk-free rate (annual)", 0.04))
    LOOKBACK = int(_ask("Years of price history", 3))

    print("\n── Portfolio weights ────────────────────────────────────")
    print("  Format : TICKER:weight, TICKER:weight  (decimals or %)")
    print("  Example: BOTZ:0.20, GRID:0.15, CIBR:0.13")
    while True:
        raw = input("  Weights: ").strip()
        BASELINE = _parse_pairs(raw)
        if BASELINE: break
        print("  ✗ Nothing parsed — try again.")
    total = sum(BASELINE.values())
    if total > 1.5: BASELINE = {k: v/100 for k, v in BASELINE.items()}; total /= 100
    if abs(total - 1.0) > 0.005:
        print(f"  Weights sum to {total:.3f} — normalising to 1.0")
        BASELINE = {k: v/total for k, v in BASELINE.items()}
    TICKERS = list(BASELINE.keys())
    print(f"  ✓ {len(TICKERS)} tickers: {', '.join(TICKERS)}")

    print("\n── Per-asset weight bounds ──────────────────────────────")
    WEIGHT_MIN = float(_ask("  Min weight per ticker", 0.00))
    WEIGHT_MAX = float(_ask("  Max weight per ticker", 0.22))

    print("\n── Target return (optional — for PPO_TargetReturn method)")
    print("  Example: 6 for 6%, 12 for 12%, press Enter to skip")
    raw_tr = input("  Target return %: ").strip()
    TARGET_RETURN = float(raw_tr) / 100 if raw_tr else None

    print("\n── Themes (optional — press Enter to skip) ──────────────")
    print("  Format : ThemeName:TICK1+TICK2, ThemeName2:TICK3")
    raw_themes = input("  Themes : ").strip()
    THEMES = {}
    if raw_themes:
        for part in raw_themes.split(","):
            part = part.strip()
            if ":" not in part: continue
            name, tks = part.split(":", 1)
            THEMES[name.strip()] = [t.strip().upper() for t in tks.split("+") if t.strip()]

    print("\n── Black-Litterman views (optional — press Enter to skip)")
    print("  Format : TICKER:expected_annual_return")
    raw_views = input("  BL views: ").strip()
    BL_VIEWS      = _parse_pairs(raw_views) if raw_views else {}
    BL_CONFIDENCE = 0.85
    BL_RELATIVE   = []
    if BL_VIEWS:
        BL_CONFIDENCE = float(_ask("  Confidence (0=history, 1=views)", 0.85))
        print("  Relative views: TICK1+TICK2>TICK3:spread  (semicolons for multiple, Enter to skip)")
        raw_rel = input("  Relative: ").strip()
        if raw_rel: BL_RELATIVE = _parse_relative(raw_rel)

    print("\n── Group weight constraints (optional — press Enter to skip)")
    print("  Format : TICK1+TICK2:min:max  (comma-separated)")
    raw_gb = input("  Groups : ").strip()
    GROUP_BOUNDS = []
    if raw_gb:
        for part in raw_gb.split(","):
            pieces = [p.strip() for p in part.split(":")]
            if len(pieces) < 3: continue
            members = [t.strip().upper() for t in pieces[0].split("+") if t.strip()]
            GROUP_BOUNDS.append((members, float(pieces[1]), float(pieces[2])))

    print("\n── Relative group constraints (optional — press Enter to skip)")
    print("  Format : TICK1+TICK2>TICK3:gap  (semicolons for multiple)")
    raw_rg = input("  Relative: ").strip()
    GROUP_RELATIVE = _parse_relative(raw_rg) if raw_rg else []

    OUTPUT_FILE = _ask("\nOutput filename", "portfolio_analysis.xlsx")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("╔══════════════════════════════════════════════════════╗")
print("║                  Running analysis…                  ║")
print("╚══════════════════════════════════════════════════════╝")
print(f"  Tickers      : {', '.join(TICKERS)}")
print(f"  BL views     : {'yes (' + str(len(BL_VIEWS)) + ' tickers)' if BL_VIEWS else 'skipped'}")
print(f"  Themes       : {'yes (' + str(len(THEMES)) + ')' if THEMES else 'skipped'}")
print(f"  Target return: {str(round(TARGET_RETURN*100,1))+'%' if TARGET_RETURN else 'not set'}")
print(f"  Constraints  : {len(GROUP_BOUNDS)} group bounds, {len(GROUP_RELATIVE)} relative")
print()

# ════════════════════════════════════════════════════════════════════════════════
#  ENGINE  ← generic, never needs editing
# ════════════════════════════════════════════════════════════════════════════════

# ── 1. Fetch prices ───────────────────────────────────────────────────────────
end   = datetime.today()
start = end - timedelta(days=int(365.25 * LOOKBACK) + 5)
print("Fetching prices from yfinance…")
raw     = yf.download(TICKERS, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=True)
prices  = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw)[TICKERS].dropna()
returns = prices.pct_change().dropna()
print(f"  {returns.shape[0]} trading days × {returns.shape[1]} tickers\n")

# Also fetch SPY for CAPM estimation
print("Fetching SPY for CAPM…")
try:
    spy_raw    = yf.download("SPY", start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    spy_prices = (spy_raw["Close"] if isinstance(spy_raw.columns, pd.MultiIndex) else spy_raw).squeeze()
    spy_prices = spy_prices.reindex(prices.index).dropna()
except Exception:
    spy_prices = None
    print("  SPY fetch failed — CAPM method will be skipped")

# ── 2. Portfolio statistics ───────────────────────────────────────────────────
def port_stats(weights, rf=RF):
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

# ── 3. Black-Litterman posterior ──────────────────────────────────────────────
def _bl_posterior(abs_views, rel_views, confidence, tau=0.05, periods=252):
    tickers = list(returns.columns)
    Sigma   = returns.cov().values * periods
    pi      = returns.mean().values * periods
    rows, q = [], []
    for t, v in abs_views.items():
        if t in tickers:
            row = np.zeros(len(tickers)); row[tickers.index(t)] = 1.0
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

# ── 4. Riskfolio constraint builder ──────────────────────────────────────────
def _rf_constraints(tickers):
    rows_A, rows_B = [], []
    for members, lo, hi in GROUP_BOUNDS:
        ind = np.array([1.0 if t in members else 0.0 for t in tickers])
        if ind.sum() == 0: continue
        rows_A.append(-ind); rows_B.append(-lo)
        rows_A.append( ind); rows_B.append( hi)
    for group_a, group_b, gap in GROUP_RELATIVE:
        a = np.array([1.0 if t in group_a else 0.0 for t in tickers])
        b = np.array([1.0 if t in group_b else 0.0 for t in tickers])
        rows_A.append(b - a); rows_B.append(-gap)
    if not rows_A: return None, None
    return np.vstack(rows_A), np.array(rows_B).reshape(-1, 1)

def _make_rf_port(mu_override=None, cov_override=None):
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="ledoit")
    if mu_override is not None: port.mu  = mu_override.to_frame().T
    if cov_override is not None: port.cov = cov_override
    port.sht      = False
    port.upperlng = WEIGHT_MAX
    A, B = _rf_constraints(list(returns.columns))
    if A is not None:
        port.ainequality = A
        port.binequality = B
    return port

def _rf_max_sharpe(port):
    best_s, best_w = -np.inf, None
    for l in np.geomspace(0.1, 200, 30):
        w = port.optimization(model="Classic", rm="MV", obj="Utility",
                              rf=RF, l=float(l), hist=True)
        if w is None: continue
        s = port_stats(w["weights"])["Sharpe"]
        if s > best_s: best_s, best_w = s, w["weights"]
    return best_w

def _rf_weights(w_df):
    if w_df is None: return None
    return w_df["weights"] if isinstance(w_df, pd.DataFrame) else w_df

# ── 5. Run Riskfolio-Lib optimizers ──────────────────────────────────────────
print("Running Riskfolio-Lib optimizers…")
rf_results = {}
base = _make_rf_port()

# Classic optimizers
for label, rm, obj in [("RF_MinVol","MV","MinRisk"), ("RF_MinCVaR","CVaR","MinRisk"),
                        ("RF_MinCDaR","CDaR","MinRisk"), ("RF_MinMAD","MAD","MinRisk")]:
    try:
        w = base.optimization(model="Classic", rm=rm, obj=obj, rf=RF, hist=True)
        if w is not None: rf_results[label] = w["weights"]
    except Exception as e:
        print(f"  {label} failed: {e}")

w = _rf_max_sharpe(base)
if w is not None: rf_results["RF_MaxSharpe"] = w

# Risk Parity
try:
    rp2 = _make_rf_port()
    w = rp2.rp_optimization(model="Classic", rm="MV", rf=RF, hist=True)
    if w is not None: rf_results["RF_RiskParity"] = w["weights"]
except Exception as e:
    print(f"  RF_RiskParity failed: {e}")

# Hierarchical methods via HCPortfolio
for label, model in [("RF_HRP","HRP"), ("RF_HERC","HERC"), ("RF_NCO","NCO")]:
    try:
        hc = rp.HCPortfolio(returns=returns)
        w  = hc.optimization(model=model, codependence="pearson", rm="MV",
                             rf=RF, linkage="ward", max_k=10, leaf_order=True)
        if w is not None: rf_results[label] = w["weights"]
    except Exception as e:
        print(f"  {label} failed: {e}")

# Black-Litterman
has_bl = bool(BL_VIEWS)
mu_bl_rf = cov_bl_rf = None
if has_bl:
    print("  Computing BL posterior…")
    mu_bl_rf, cov_bl_rf = _bl_posterior(BL_VIEWS, BL_RELATIVE, BL_CONFIDENCE)
    try:
        bl_port = _make_rf_port(mu_override=mu_bl_rf, cov_override=cov_bl_rf)
        w = _rf_max_sharpe(bl_port)
        if w is not None: rf_results["RF_BL_MaxSharpe"] = w
    except Exception as e:
        print(f"  RF_BL_MaxSharpe failed: {e}")

print(f"  Done — {len(rf_results)} methods\n")

# ── 6. Run PyPortfolioOpt optimizers ──────────────────────────────────────────
from pypfopt import (EfficientFrontier, EfficientSemivariance, EfficientCVaR,
                     HRPOpt, risk_models, expected_returns, BlackLittermanModel)

print("Running PyPortfolioOpt optimizers…")
ppo_results = {}

def _ppo_ef(mu, S):
    """EfficientFrontier with group constraints applied."""
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

def _ppo_cvar_ef(mu, rets):
    """EfficientCVaR with group constraints."""
    ec = EfficientCVaR(mu, rets, weight_bounds=(WEIGHT_MIN, WEIGHT_MAX))
    tickers = list(rets.columns)
    for members, lo, hi in GROUP_BOUNDS:
        idx = [tickers.index(t) for t in members if t in tickers]
        if not idx: continue
        ec.add_constraint(lambda w, i=idx, lo=lo: sum(w[j] for j in i) - lo)
        ec.add_constraint(lambda w, i=idx, hi=hi: hi - sum(w[j] for j in i))
    return ec

def _ppo_weights(raw):
    w = pd.Series({t: float(raw.get(t, 0.0)) for t in TICKERS})
    s = w.sum()
    return (w / s).round(4) if s > 0 else w

try:
    S_hist  = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    mu_hist = expected_returns.mean_historical_return(prices)
    mu_ema  = expected_returns.ema_historical_return(prices)   # more weight to recent data

    # CAPM expected returns using SPY as market factor
    mu_capm = None
    if spy_prices is not None:
        try:
            mu_capm = expected_returns.capm_return(
                prices, market_prices=spy_prices, risk_free_rate=RF)
        except Exception as e:
            print(f"  CAPM return estimation failed: {e}")

    # 1. MaxSharpe — historical mean returns
    try:
        ef = _ppo_ef(mu_hist, S_hist)
        ef.max_sharpe(risk_free_rate=RF)
        ppo_results["PPO_MaxSharpe"] = _ppo_weights(ef.clean_weights())
    except Exception as e: print(f"  PPO_MaxSharpe failed: {e}")

    # 2. MinVol
    try:
        ef = _ppo_ef(mu_hist, S_hist)
        ef.min_volatility()
        ppo_results["PPO_MinVol"] = _ppo_weights(ef.clean_weights())
    except Exception as e: print(f"  PPO_MinVol failed: {e}")

    # 3. MinCVaR — minimise tail loss directly (uses historical scenarios)
    try:
        ec = _ppo_cvar_ef(mu_hist, returns)
        ec.min_cvar()
        ppo_results["PPO_MinCVaR"] = _ppo_weights(ec.clean_weights())
    except Exception as e: print(f"  PPO_MinCVaR failed: {e}")

    # 4. MinSemivariance — only penalises downside vol (optimises for Sortino)
    try:
        es = EfficientSemivariance(mu_hist, returns,
                                   weight_bounds=(WEIGHT_MIN, WEIGHT_MAX))
        es.min_semivariance()
        ppo_results["PPO_MinSemivar"] = _ppo_weights(es.clean_weights())
    except Exception as e: print(f"  PPO_MinSemivar failed: {e}")

    # 5. HRP — Hierarchical Risk Parity (clustering-based, no constraints)
    try:
        hrp = HRPOpt(returns)
        hrp.optimize()
        ppo_results["PPO_HRP"] = _ppo_weights(hrp.clean_weights())
    except Exception as e: print(f"  PPO_HRP failed: {e}")

    # 6. EMA MaxSharpe — recent returns weighted more heavily
    try:
        ef = _ppo_ef(mu_ema, S_hist)
        ef.max_sharpe(risk_free_rate=RF)
        ppo_results["PPO_EMA_MaxSharpe"] = _ppo_weights(ef.clean_weights())
    except Exception as e: print(f"  PPO_EMA_MaxSharpe failed: {e}")

    # 7. CAPM MaxSharpe — market-beta-based return estimates
    if mu_capm is not None:
        try:
            ef = _ppo_ef(mu_capm, S_hist)
            ef.max_sharpe(risk_free_rate=RF)
            ppo_results["PPO_CAPM_MaxSharpe"] = _ppo_weights(ef.clean_weights())
        except Exception as e: print(f"  PPO_CAPM_MaxSharpe failed: {e}")

    # 8. BL MaxSharpe — PyPortfolioOpt BL with idzorek confidence weighting
    if has_bl:
        try:
            abs_u = {k: v for k, v in BL_VIEWS.items() if k in TICKERS}
            bl_m  = BlackLittermanModel(S_hist, pi=mu_hist,
                                        absolute_views=abs_u, omega="idzorek",
                                        view_confidences=[BL_CONFIDENCE]*len(abs_u))
            ef = _ppo_ef(bl_m.bl_returns(), bl_m.bl_cov())
            ef.max_sharpe(risk_free_rate=RF)
            ppo_results["PPO_BL_MaxSharpe"] = _ppo_weights(ef.clean_weights())
        except Exception as e: print(f"  PPO_BL_MaxSharpe failed: {e}")

    # 9. TargetReturn — minimum vol portfolio hitting your target (if set)
    if TARGET_RETURN:
        try:
            ef = _ppo_ef(mu_hist, S_hist)
            ef.efficient_return(target_return=TARGET_RETURN)
            label = f"PPO_Target_{int(TARGET_RETURN*100)}pct"
            ppo_results[label] = _ppo_weights(ef.clean_weights())
        except Exception as e: print(f"  PPO_TargetReturn failed: {e}")

except Exception as e:
    print(f"  PyPortfolioOpt setup failed: {e}")

print(f"  Done — {len(ppo_results)} methods\n")

# ── 7. Combine results ────────────────────────────────────────────────────────
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

# Return estimators comparison sheet
return_ests = pd.DataFrame({"mu_hist": expected_returns.mean_historical_return(prices),
                             "mu_ema":  expected_returns.ema_historical_return(prices)})
if mu_capm is not None: return_ests["mu_capm"] = mu_capm
if has_bl and mu_bl_rf is not None: return_ests["mu_BL"] = mu_bl_rf
return_ests = return_ests.round(4)

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
    weights_df.to_excel(xw,     sheet_name="Weights")
    dollar_df.to_excel(xw,      sheet_name="Dollars")
    stats_df[RISK_COLS].to_excel(xw, sheet_name="RiskCompare")
    stats_df.to_excel(xw,       sheet_name="Stats_Full")
    if not theme_df.empty:
        theme_df.to_excel(xw,   sheet_name="ThemeExposure")
    return_ests.to_excel(xw,    sheet_name="ReturnEstimates")

    # Method legend sheet
    legend = pd.DataFrame([
        # Riskfolio
        ("RF_MaxSharpe",        "Riskfolio", "Utility sweep maximising Sharpe ratio"),
        ("RF_MinVol",           "Riskfolio", "Minimum variance (MV)"),
        ("RF_MinCVaR",          "Riskfolio", "Minimum Conditional Value-at-Risk (tail loss)"),
        ("RF_MinCDaR",          "Riskfolio", "Minimum Conditional Drawdown-at-Risk"),
        ("RF_MinMAD",           "Riskfolio", "Minimum Mean Absolute Deviation"),
        ("RF_RiskParity",       "Riskfolio", "Equal risk contribution across assets"),
        ("RF_HRP",              "Riskfolio", "Hierarchical Risk Parity (clustering)"),
        ("RF_HERC",             "Riskfolio", "Hierarchical Equal Risk Contribution"),
        ("RF_NCO",              "Riskfolio", "Nested Clustered Optimisation"),
        ("RF_BL_MaxSharpe",     "Riskfolio", "Black-Litterman posterior → MaxSharpe"),
        # PyPortfolioOpt
        ("PPO_MaxSharpe",       "PyPortfolioOpt", "Max Sharpe using historical mean returns"),
        ("PPO_MinVol",          "PyPortfolioOpt", "Minimum volatility"),
        ("PPO_MinCVaR",         "PyPortfolioOpt", "Minimum CVaR (historical scenarios)"),
        ("PPO_MinSemivar",      "PyPortfolioOpt", "Minimum semivariance (downside vol only)"),
        ("PPO_HRP",             "PyPortfolioOpt", "Hierarchical Risk Parity (no constraints)"),
        ("PPO_EMA_MaxSharpe",   "PyPortfolioOpt", "MaxSharpe using EMA returns (recent-weighted)"),
        ("PPO_CAPM_MaxSharpe",  "PyPortfolioOpt", "MaxSharpe using CAPM return estimates"),
        ("PPO_BL_MaxSharpe",    "PyPortfolioOpt", "Black-Litterman (idzorek) → MaxSharpe"),
        ("PPO_Target_Xpct",     "PyPortfolioOpt", "Min-vol portfolio hitting target return X%"),
    ], columns=["Method", "Library", "Description"])
    legend.to_excel(xw, sheet_name="MethodLegend", index=False)

print(f"\nExcel written → {out.resolve()}")

# ── 9. Frontier plot ──────────────────────────────────────────────────────────
COLORS = [
    "#d62728","#1f77b4","#2ca02c","#9467bd","#ff7f0e","#17becf",
    "#e377c2","#8c564b","#bcbd22","#7f7f7f","#aec7e8","#ffbb78",
    "#98df8a","#ff9896","#c5b0d5","#c49c94","#f7b6d2","#dbdb8d",
    "#9edae5","#393b79",
]
fig, ax = plt.subplots(figsize=(13, 8))
try:
    fport    = _make_rf_port()
    frontier = fport.efficient_frontier(model="Classic", rm="MV",
                                        points=30, rf=RF, hist=True)
    fs = [port_stats(frontier[c]) for c in frontier.columns]
    ax.plot([s["AnnVol"] for s in fs], [s["AnnReturn"] for s in fs],
            "-", color="#ddd", lw=2, label="MV Frontier", zorder=1)
except Exception:
    pass

for col, color in zip(weights_df.columns, COLORS):
    s = port_stats(weights_df[col])
    prefix = "★ " if col == "Baseline" else ""
    ax.scatter(s["AnnVol"], s["AnnReturn"], s=150, color=color,
               edgecolor="black", zorder=5, label=f"{prefix}{col}")

ax.set_xlabel("Annualised Volatility")
ax.set_ylabel("Annualised Return")
ax.set_title("Portfolio Comparison — Real Market Data")
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="upper left", ncol=2)
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
