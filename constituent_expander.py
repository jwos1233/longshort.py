"""
Constituent Expander – ETF weights → stock + non-equity weights
================================================================

Takes target weights at the ETF level and, for each equity ETF with constituent
data, applies the same methodology (volatility chasing + EMA filter) to its
constituents, then splits that ETF's weight over the top N performers.

Non-equity ETFs (bonds, commodities, crypto) and unmapped ETFs are left as-is.

Usage:
    from constituent_expander import expand_etf_weights_to_constituents

    # After signal generator produces etf_weights (e.g. QQQ=0.34, XLE=0.15, GLD=0.08)
    stock_weights = expand_etf_weights_to_constituents(
        etf_weights, price_data,
        vol_lookback=30, ema_period=50, top_n_per_etf=5
    )
    # Result: { 'NVDA': 0.08, 'MSFT': 0.07, ... (QQQ split), 'GLD': 0.08 (unchanged), ... }
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from etf_mapper import (
    ETF_CONSTITUENTS,
    NON_EQUITY_ETFS,
    UNMAPPED_EQUITY_ETFS,
    is_equity_etf_mapped,
)


def expand_etf_weights_to_constituents(
    etf_weights: Dict[str, float],
    price_data: pd.DataFrame,
    vol_lookback: int = 30,
    ema_period: int = 50,
    top_n_per_etf: int = 5,
    as_of_date: Optional[pd.Timestamp] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Expand ETF target weights to constituent-level weights for equity ETFs.

    For each equity ETF with constituent data:
      - Get constituents that exist in price_data.
      - Compute rolling vol and EMA up to as_of_date (or last row).
      - Apply EMA filter (price > EMA).
      - Rank by volatility (vol chasing), take top_n_per_etf.
      - Split that ETF's weight among the top N by volatility (vol-weighted).

    Non-equity and unmapped ETFs keep their weight on the ETF ticker.

    Args:
        etf_weights: {etf_ticker: weight} from strategy (e.g. calculate_target_weights).
        price_data: DataFrame with both ETF and constituent tickers (index=date).
        vol_lookback: Rolling window for annualized vol (default 30).
        ema_period: EMA span for filter (default 50).
        top_n_per_etf: Number of top constituents per ETF (default 5).
        as_of_date: Date to use for price/vol/EMA (default: last index of price_data).
        verbose: If True, print which ETFs were expanded.

    Returns:
        Dict[ticker, weight] — mix of stock tickers and non-equity ETF tickers.
    """
    if not etf_weights:
        return {}

    if as_of_date is None:
        as_of_date = price_data.index[-1]

    # Slice history up to as_of_date (no look-ahead)
    hist = price_data.loc[price_data.index <= as_of_date]
    if len(hist) < max(vol_lookback, ema_period) + 1:
        # Not enough history; return ETF weights unchanged
        return dict(etf_weights)

    returns = hist.pct_change()
    vol_series = returns.rolling(window=vol_lookback).std() * np.sqrt(252)
    ema_series = hist.ewm(span=ema_period, adjust=False).mean()

    # Use last row of history for "current" vol and EMA
    last_idx = hist.index[-1]
    vols = vol_series.loc[last_idx] if last_idx in vol_series.index else vol_series.iloc[-1]
    ema = ema_series.loc[last_idx] if last_idx in ema_series.index else ema_series.iloc[-1]
    prices = hist.loc[last_idx] if last_idx in hist.index else hist.iloc[-1]

    result: Dict[str, float] = {}
    expanded_etfs = []

    for etf, weight in etf_weights.items():
        if weight <= 0:
            continue

        # Non-equity or unmapped: keep ETF weight
        if etf in NON_EQUITY_ETFS or etf in UNMAPPED_EQUITY_ETFS or not is_equity_etf_mapped(etf):
            result[etf] = result.get(etf, 0.0) + weight
            continue

        constituents = ETF_CONSTITUENTS[etf]
        # Only use constituents we have price data for
        available = [c for c in constituents if c in price_data.columns]
        if not available:
            # No data for any constituent; keep ETF weight
            result[etf] = result.get(etf, 0.0) + weight
            continue

        # Vol and EMA for available constituents
        quad_vols = {}
        for ticker in available:
            if ticker not in vols.index or pd.isna(vols[ticker]) or vols[ticker] <= 0:
                continue
            price = prices[ticker] if ticker in prices.index else np.nan
            ema_val = ema[ticker] if ticker in ema.index else np.nan
            if pd.isna(price) or pd.isna(ema_val) or price <= ema_val:
                continue
            quad_vols[ticker] = float(vols[ticker])

        if not quad_vols:
            # No constituent passed EMA; keep weight on ETF
            result[etf] = result.get(etf, 0.0) + weight
            continue

        # Top N by volatility (vol chasing)
        sorted_const = sorted(quad_vols.items(), key=lambda x: x[1], reverse=True)[:top_n_per_etf]
        total_vol = sum(v for _, v in sorted_const)
        if total_vol <= 0:
            result[etf] = result.get(etf, 0.0) + weight
            continue

        expanded_etfs.append(etf)
        for ticker, vol in sorted_const:
            w = weight * (vol / total_vol)
            result[ticker] = result.get(ticker, 0.0) + w

    if verbose and expanded_etfs:
        print(f"  Expanded to constituents: {', '.join(expanded_etfs)}")

    return result
