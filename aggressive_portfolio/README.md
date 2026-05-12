# Aggressive Thematic Portfolio — Riskfolio Analysis

Benchmarks the v4 intuition-built thematic portfolio (AI infra, defense/aerospace,
nuclear, robotics, plus quantum/cyber/biotech kickers) against four classical
optimizers and a Black-Litterman run that overrides naive historical means with
forward views.

## Quick start

```bash
cd aggressive_portfolio
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m aggressive_portfolio
```

Outputs land in `outputs/`:
- `portfolio_comparison.xlsx` — weights, stats, theme exposure, expected returns
- `efficient_frontier.png` — frontier with each portfolio plotted as a point

## What it runs

| Method | Objective | Why it's here |
| --- | --- | --- |
| `v4` | (baseline) | Your intuition-weighted portfolio |
| `MaxSharpe_MV` | Max Sharpe under MV | What history says is optimal |
| `MinVol_MV` | Lowest variance | Risk-floor reference |
| `MinCVaR` | Min tail loss (95%) | Honest downside view |
| `RiskParity` | Equal risk contribution | Tells you which holdings dominate risk |
| `BlackLitterman_MaxSharpe` | Max Sharpe with BL views | History + your conviction |

## Forward views (edit in `config.py`)

Absolute (annualized expected return):
- QTUM 8% (cap the 86% extrapolation), XBI 12%, BOTZ 14%, ARKQ/SYM 13%,
  URA 13%, GRID 12%, CIBR 10%, SMH 9%.

Relative:
- Aerospace (XAR, RKLB) outperforms Defense (SHLD, KTOS) by 4%.

## Constraints

- Long-only, weights sum to 1.
- Per-asset bounds: 1% ≤ w ≤ 22% (BOTZ cap).
- Robotics 20–25%, Nuclear 10–20%, Fixed Income 5–20%.
- Aerospace ≥ Defense (group inequality).

## Layout

```
aggressive_portfolio/
├── config.py              # weights, views, constraints
├── data_loader.py         # yfinance + CSV fallback
├── constraints.py         # builds A, B, bounds for Riskfolio
├── black_litterman.py     # He-Litterman posterior
├── optimize.py            # 4 optimizers + BL + portfolio_stats
├── compare.py             # tables, Excel, frontier plot
├── __main__.py            # entrypoint
├── data/                  # cached price CSVs
└── outputs/               # generated artifacts
```

## Reading the results

The point of this isn't to mechanically adopt the optimizer's weights. It's to
locate the **disagreement** between intuition and math:

- If MaxSharpe_MV wants BOTZ at 12% and v4 has it at 22%, that 10pt is a
  conviction premium — defensible if you believe the future ≠ the lookback.
- If RiskParity says URA + BOTZ contribute most of the portfolio risk, decide
  whether that concentration is rewarded or just stomach-churn.
- If BlackLitterman_MaxSharpe (which already incorporates your views) still
  diverges from v4, the views are too modest or the constraints are binding.

## Caveats

- yfinance "Adj Close" handling differs across versions; the loader prefers
  `Close` from the auto-adjusted feed.
- ETFs with short history (SHLD, AVUV peers) get truncated by `dropna()` —
  consider relaxing the lookback if a ticker drops out.
- Historical mean is a poor forward estimate for assets coming off a 100%+
  year. The Black-Litterman path is the methodologically honest one.
