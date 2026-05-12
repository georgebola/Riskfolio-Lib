"""Run end-to-end:  python -m aggressive_portfolio

Pulls (or loads) prices, runs all optimizers, writes Excel + frontier plot.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .black_litterman import posterior
from .compare import (
    plot_efficient_frontier,
    stats_table,
    theme_exposure,
    weights_table,
    write_excel,
)
from .config import CONSTRAINTS, LOOKBACK_YEARS, TICKERS, V4_WEIGHTS
from .data_loader import load_prices, to_returns
from .optimize import run_all

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    log = logging.getLogger("aggressive_portfolio")

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    log.info("Loading prices for %d tickers, %d-year lookback", len(TICKERS), LOOKBACK_YEARS)
    prices = load_prices(TICKERS, lookback_years=LOOKBACK_YEARS)
    returns = to_returns(prices).dropna()
    log.info("Returns frame: %s", returns.shape)

    mu_bl, cov_bl = posterior(returns)
    log.info("Black-Litterman mu (annualized):\n%s", mu_bl.round(3).to_string())

    results = run_all(returns, CONSTRAINTS, mu_bl=mu_bl, cov_bl=cov_bl)
    weights = weights_table(results, V4_WEIGHTS)
    stats = stats_table(weights, returns)
    themes = theme_exposure(weights)

    log.info("\n=== Weights ===\n%s", weights.to_string())
    log.info("\n=== Stats ===\n%s", stats.to_string())
    log.info("\n=== Theme exposure ===\n%s", themes.to_string())

    mu_hist = returns.mean() * 252
    write_excel(
        out_dir / "portfolio_comparison.xlsx",
        weights,
        stats,
        themes,
        mu_hist,
        mu_bl,
    )
    plot_efficient_frontier(returns, weights, out_dir / "efficient_frontier.png")


if __name__ == "__main__":
    main()
