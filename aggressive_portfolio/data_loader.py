"""Price/return data loading with yfinance + on-disk CSV fallback.

Strategy:
1. Try yfinance with the requested lookback window.
2. On any failure (offline, rate limit, missing ticker), fall back to a CSV
   cached from the most recent successful run.
3. Always persist a fresh CSV after a successful yfinance pull so the cache
   stays current.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.csv"


def _fetch_yfinance(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so the module loads without the dep

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # yfinance returns either a flat frame (one ticker) or a MultiIndex.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        elif "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"]
        else:
            raise RuntimeError("yfinance returned no Close/Adj Close columns")
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise RuntimeError(f"yfinance returned no data for: {missing}")
    if len(prices) == 0:
        raise RuntimeError("yfinance returned 0 rows for all tickers")
    return prices[tickers]


def load_prices(
    tickers: List[str],
    lookback_years: int = 3,
    end: Optional[str] = None,
    cache_name: str = "prices",
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices indexed by date."""

    end_dt = datetime.utcnow() if end is None else datetime.fromisoformat(end)
    start_dt = end_dt - timedelta(days=int(365.25 * lookback_years) + 5)
    start = start_dt.date().isoformat()
    end_str = end_dt.date().isoformat()

    cache = _cache_path(cache_name)

    try:
        prices = _fetch_yfinance(tickers, start, end_str)
        prices.to_csv(cache)
        logger.info("Fetched %d rows from yfinance, cached to %s", len(prices), cache)
        return prices
    except Exception as exc:  # noqa: BLE001 — broad fallback by design
        logger.warning("yfinance fetch failed (%s); falling back to cache %s", exc, cache)
        if not cache.exists():
            raise RuntimeError(
                f"yfinance failed and no cache at {cache}. Run once with network access."
            ) from exc
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise RuntimeError(f"Cache is missing tickers: {missing}") from exc
        return prices[tickers]


def to_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    """Convert a price frame to daily returns."""
    if kind == "simple":
        return prices.pct_change().dropna(how="all")
    if kind == "log":
        import numpy as np

        return (prices.apply(np.log)).diff().dropna(how="all")
    raise ValueError(f"Unknown return kind: {kind}")
