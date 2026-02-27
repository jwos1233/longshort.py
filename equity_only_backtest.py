"""
Equity-only variant of the Macro Quadrant backtest.

This subclass mirrors the production QuadrantPortfolioBacktest logic but:
  - Only allocates risk to *equity* ETFs / stocks.
  - Any allocation to NON_EQUITY_ETFS (bonds, commodities, crypto, vol, etc.)
    is treated as CASH (no trade).
"""

from __future__ import annotations

import pandas as pd

from config import QUAD_ALLOCATIONS
from etf_mapper import NON_EQUITY_ETFS
from quad_portfolio_backtest import (
    QuadrantPortfolioBacktest,
    BASE_QUAD_LEVERAGE,
    Q1_LEVERAGE_MULTIPLIER,
)


class EquityOnlyBacktest(QuadrantPortfolioBacktest):
    """
    Backtest variant that only allocates risk to *equity* ETFs / stocks.

    Any allocation to tickers in NON_EQUITY_ETFS (bonds, commodities, crypto,
    volatility, etc.) is treated as CASH:
      - That portion of the quad sleeve is left uninvested.
      - The invested part of the quad is scaled by the equity share of
        QUAD_ALLOCATIONS for that quad.
    """

    def __init__(self, *args, **kwargs):
        # Delegate construction to the base class, then compute per-quad
        # equity fractions used when building target weights.
        super().__init__(*args, **kwargs)
        self._quad_equity_fraction = {}
        for quad, alloc in QUAD_ALLOCATIONS.items():
            total = float(sum(alloc.values())) if alloc else 0.0
            if total <= 0:
                self._quad_equity_fraction[quad] = 0.0
                continue
            equity_total = sum(
                w for t, w in alloc.items()
                if t not in NON_EQUITY_ETFS
            )
            self._quad_equity_fraction[quad] = equity_total / total if total > 0 else 0.0

    def calculate_target_weights(self, top_quads: pd.DataFrame) -> pd.DataFrame:
        """
        Same macro regime logic as the base class, but:
          - Only include equity ETFs (exclude NON_EQUITY_ETFS).
          - Scale each quad's effective leverage by its equity share so the
            non-equity sleeves remain as cash.
        """
        weights = pd.DataFrame(0.0, index=top_quads.index,
                               columns=self.price_data.columns)

        for date in top_quads.index:
            top1 = top_quads.loc[date, "Top1"]
            top2 = top_quads.loc[date, "Top2"]
            final_weights: dict[str, float] = {}

            for quad in (top1, top2):
                # Base leverage (same as production)
                base_quad_weight = BASE_QUAD_LEVERAGE
                if quad == "Q1":
                    base_quad_weight *= Q1_LEVERAGE_MULTIPLIER

                # Fraction of this quad sleeve that is real equity
                equity_frac = self._quad_equity_fraction.get(quad, 1.0)
                if equity_frac <= 0:
                    # No equity exposure for this quad -> all cash
                    continue

                quad_weight = base_quad_weight * equity_frac

                # Only equity ETFs in this quad, present in price_data
                quad_tickers = [
                    t for t in QUAD_ALLOCATIONS[quad].keys()
                    if t in self.price_data.columns and t not in NON_EQUITY_ETFS
                ]
                if not quad_tickers:
                    continue

                quad_vols = {}
                for ticker in quad_tickers:
                    if ticker in self.volatility_data.columns and date in self.volatility_data.index:
                        vol = self.volatility_data.loc[date, ticker]
                        if pd.notna(vol) and vol > 0:
                            quad_vols[ticker] = vol

                if not quad_vols:
                    continue

                total_vol = sum(quad_vols.values())
                if total_vol <= 0:
                    continue

                vol_weights = {
                    t: (v / total_vol) * quad_weight
                    for t, v in quad_vols.items()
                }

                # Apply EMA filter; any ticker failing the filter stays as cash.
                for ticker, weight in vol_weights.items():
                    if ticker in self.ema_data.columns and date in self.price_data.index:
                        price = self.price_data.loc[date, ticker]
                        ema = self.ema_data.loc[date, ticker]
                        if pd.notna(price) and pd.notna(ema) and price > ema:
                            final_weights[ticker] = final_weights.get(ticker, 0.0) + weight

            # Note: BTC-USD and other NON_EQUITY_ETFS are implicitly treated as
            # cash here; we do not apply BTC_PROXY_BASKET or other overrides.

            # Respect max_positions if set (same logic as base class)
            if self.max_positions and len(final_weights) > self.max_positions:
                sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
                top_n_weights = dict(sorted_weights[: self.max_positions])
                original_total = sum(final_weights.values())
                new_total = sum(top_n_weights.values())
                scale_factor = original_total / new_total if new_total > 0 else 1.0
                final_weights = {t: w * scale_factor for t, w in top_n_weights.items()}

            for ticker, weight in final_weights.items():
                weights.loc[date, ticker] = weight

        return weights

