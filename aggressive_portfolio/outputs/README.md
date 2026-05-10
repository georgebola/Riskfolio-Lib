# Outputs

These artifacts were generated from **synthetic price data** because the
sandbox where the package was first run blocks yfinance. The synthetic series
(see `gen_synth.py` if checked in, otherwise reconstruct from `config.py`
volatilities) match each ticker's realistic annual vol and beta to a market
factor — they are useful to verify the pipeline runs end-to-end, but the
weights and stats are NOT real recommendations.

To regenerate with real data, run on your machine where yfinance has network
access:

```bash
rm aggressive_portfolio/data/prices.csv  # force a fresh fetch
python -m aggressive_portfolio
```

## Files

- `portfolio_comparison.xlsx` — sheets:
  - `Weights`: v4 plus 5 optimizer outputs and a `v4_minus_optMaxSharpe` delta
  - `Stats`: AnnReturn, AnnVol, Sharpe, daily VaR/CVaR(95%) per portfolio
  - `ThemeExposure`: look-through weight by theme for every portfolio
  - `ExpectedReturns`: historical mean vs. Black-Litterman posterior mu

- `efficient_frontier.png` — long-only MV frontier with each portfolio plotted

## Known issue: Max Sharpe workaround

riskfolio 7.2.1 + cvxpy 1.8.2 fails the Charnes-Cooper Sharpe optimization
("problem doesn't have a solution"). `optimize._max_sharpe_via_utility_sweep`
sweeps the risk-aversion `l` in `obj='Utility'` and picks the empirical-Sharpe
maximum as a robust workaround. If a future riskfolio release fixes the bug,
swap back to the direct `obj='Sharpe'` call in `_opt`.
