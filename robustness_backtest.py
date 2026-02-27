"""
Robustness backtest: Equity-only variant with top P/L tickers excluded.

Excludes a configurable list of tickers (e.g. top 10 by P/L from a prior run)
to test whether returns are robust or concentrated in a few names.
"""

from __future__ import annotations

import pandas as pd

from equity_only_backtest import EquityOnlyBacktest


# Default: top 10 P/L tickers from equity-only backtest (update after each run)
EXCLUDED_TOP10_DEFAULT = [
    "BMNR",
    "SGML",
    "LAC",
    "UUUU",
    "RGTI",
    "LTBR",
    "ALB",
    "CRWV",
    "OKLO",
    "MP",
]


class ExcludeTopRobustnessBacktest(EquityOnlyBacktest):
    """
    Equity-only backtest with specified tickers excluded from the universe.

    Use this to test robustness: exclude top P/L contributors and see how
    the strategy performs without them.
    """

    def __init__(self, exclude_tickers=None, **kwargs):
        super().__init__(**kwargs)
        self.exclude_tickers = set(exclude_tickers or [])

    def _filter_target_weights_before_simulation(self, target_weights: pd.DataFrame) -> pd.DataFrame:
        """Zero out excluded tickers after ETF expansion (where they get reintroduced)."""
        tw = target_weights.copy()
        for t in self.exclude_tickers:
            if t in tw.columns:
                tw[t] = 0.0
        return tw

    def print_ticker_attribution(self, top_n: int = 25):
        """
        Print full-period P/L attribution by ticker for the robustness run.

        Uses daily_ticker_pnl accumulated during run_backtest().
        """
        if not getattr(self, "daily_ticker_pnl", None):
            print("\n[Attribution] No daily_ticker_pnl data available.")
            return

        df = pd.DataFrame(self.daily_ticker_pnl)
        grouped = (
            df.groupby("ticker")["pnl_pct"]
            .sum()
            .sort_values(ascending=False)
        )

        print("\n" + "=" * 70)
        print("FULL-PERIOD P/L ATTRIBUTION BY TICKER")
        print("=" * 70)
        print(f"{'Rank':<6}{'Ticker':<15}{'P/L (pct pts)':>15}")
        print("-" * 70)

        for rank, (ticker, pnl) in enumerate(grouped.head(top_n).items(), 1):
            print(f"{rank:<6}{ticker:<15}{pnl:>14.4f}")

        print("=" * 70)
