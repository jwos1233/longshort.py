"""
Equity-Only Live Trader - Single Run
====================================

Runs the equity-only Macro Quadrant strategy live:
1. Generate signals from yesterday's close (T-1)
2. Confirm with today's EMA (T)
3. Execute trades immediately on IBKR account U24651836 (configurable via IB_ACCOUNT)

Usage:
    # Dry run (see what it would do)
    python live_trader_equity_only.py --port 4001

    # Live execution
    python live_trader_equity_only.py --port 4001 --live

    # Disable Telegram
    python live_trader_equity_only.py --port 4001 --live --no-telegram
"""

import argparse
from datetime import datetime

import pandas as pd
import yfinance as yf

from signal_generator import SignalGenerator
from ib_executor import IBExecutor
from position_manager import PositionManager
from telegram_notifier import get_notifier
from config import IB_ACCOUNT
from etf_mapper import NON_EQUITY_ETFS


class EquityOnlyLiveTrader:
    """Live trader for the equity-only Macro Quadrant strategy."""

    def __init__(self, ib_port=4001, dry_run=True, enable_telegram=True, ib_account=None):
        """
        Initialize trader

        Args:
            ib_port: IB Gateway port (4001=live, 4002=paper)
            dry_run: If True, show what would happen but don't trade
            enable_telegram: Send notifications
            ib_account: IBKR account ID (overrides IB_ACCOUNT from config/env)
        """
        self.signal_gen = SignalGenerator(
            momentum_days=20,
            ema_period=50,
            vol_lookback=30,
            max_positions=10,
            atr_stop_loss=2.0,
            atr_period=14,
            ema_smoothing_period=20,  # EMA smoothing for quad scores
        )
        self.ib_port = ib_port
        self.dry_run = dry_run
        self.ib_account = (ib_account or IB_ACCOUNT or "").strip()
        self.telegram = get_notifier() if enable_telegram else None

    def _apply_equity_only_filter(self, target_weights: dict) -> dict:
        """
        Remove non-equity sleeves (bonds, commodities, crypto, vol) and
        rescale remaining equity weights to preserve total leverage.
        """
        if not target_weights:
            return {}

        eq_weights = {t: w for t, w in target_weights.items() if t not in NON_EQUITY_ETFS}
        if not eq_weights:
            return {}

        total_orig = sum(target_weights.values())
        total_eq = sum(eq_weights.values())
        if total_orig > 0 and total_eq > 0:
            scale = total_orig / total_eq
            eq_weights = {t: w * scale for t, w in eq_weights.items()}

        return eq_weights

    def get_current_ema_status(self, tickers: list, ema_period: int = 50) -> dict:
        """
        Get CURRENT EMA status for confirmation (same as live_trader_simple).
        """
        print(f"\nFetching current market data for {len(tickers)} tickers...")

        price_data = yf.download(
            tickers,
            period="3mo",
            progress=False,
            auto_adjust=True,
        )["Close"]

        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame(name=tickers[0])

        print(f"+ Loaded data for {len(price_data.columns)} tickers")

        ema_status = {}
        for ticker in tickers:
            if ticker not in price_data.columns:
                print(f"  WARNING: No data for {ticker}")
                continue

            prices = price_data[ticker].dropna()
            if len(prices) < ema_period:
                print(f"  WARNING: Not enough data for {ticker}")
                continue

            ema = prices.ewm(span=ema_period, adjust=False).mean()
            current_price = prices.iloc[-1]
            current_ema = ema.iloc[-1]
            is_above = current_price > current_ema

            ema_status[ticker] = {
                "current_price": current_price,
                "current_ema": current_ema,
                "is_above_ema": is_above,
            }

        return ema_status

    def confirm_entries(self, target_weights: dict, ema_status: dict) -> tuple[dict, dict]:
        """
        Confirm entries using current EMA (copied from live_trader_simple).
        """
        print("\n" + "=" * 70)
        print(f"ENTRY CONFIRMATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print(f"Pending entries: {len(target_weights)}\n")

        confirmed = {}
        rejected = {}

        for ticker, weight in target_weights.items():
            if ticker not in ema_status:
                print(f"  WARNING: {ticker}: No EMA data - SKIP")
                rejected[ticker] = weight
                continue

            status = ema_status[ticker]
            current_price = status["current_price"]
            current_ema = status["current_ema"]
            is_above = status["is_above_ema"]

            if is_above:
                print(f"  ✅ {ticker}: ${current_price:.2f} > ${current_ema:.2f} EMA - CONFIRMED")
                confirmed[ticker] = weight
            else:
                print(f"  ❌ {ticker}: ${current_price:.2f} < ${current_ema:.2f} EMA - REJECTED")
                rejected[ticker] = weight

        total = len(confirmed) + len(rejected)
        rejection_rate = (len(rejected) / total * 100) if total > 0 else 0

        print("\n" + "=" * 70)
        print("CONFIRMATION SUMMARY")
        print("=" * 70)
        print(f"Confirmed: {len(confirmed)}")
        print(f"Rejected:  {len(rejected)}")
        print(f"Rejection rate: {rejection_rate:.1f}%")
        print("=" * 70)

        return confirmed, rejected

    def run(self):
        """Execute complete equity-only trading workflow."""
        print("\n" + "=" * 70)
        print(f"EQUITY-ONLY LIVE TRADER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE TRADING'}")
        print(f"Port: {self.ib_port}")
        print(f"IB Account: {self.ib_account or '(not set)'}")
        print("=" * 70)

        try:
            # STEP 1: Generate signals from YESTERDAY's close (T-1)
            print("\n" + "=" * 70)
            print("STEP 1: SIGNAL GENERATION (from yesterday's close)")
            print("=" * 70)

            signals = self.signal_gen.generate_signals()

            target_weights = signals["target_weights"]
            regime = signals["current_regime"]
            top_quads = signals["top_quadrants"]
            leverage = signals["total_leverage"]

            print(f"\nSignals generated:")
            print(f"  Regime: {regime}")
            print(f"  Top quadrants: {top_quads[0]}, {top_quads[1]}")
            print(f"  Positions (raw): {len(target_weights)}")
            print(f"  Leverage (raw): {leverage:.2f}x")

            # Apply equity-only filter
            target_weights = self._apply_equity_only_filter(target_weights)
            print(f"  Positions (equity-only): {len(target_weights)}")

            if not target_weights:
                print("\nNo equity-only positions to enter - exiting")
                return

            # STEP 2: Confirm entries using CURRENT EMA (T)
            print("\n" + "=" * 70)
            print("STEP 2: EMA CONFIRMATION (using current/today's EMA)")
            print("=" * 70)

            tickers = list(target_weights.keys())
            ema_status = self.get_current_ema_status(tickers)

            confirmed_weights, rejected = self.confirm_entries(target_weights, ema_status)

            if not confirmed_weights:
                print("\nNo entries confirmed - all rejected!")
                if self.telegram:
                    self.telegram.send_error_alert(
                        "No entries confirmed - all rejected", "Equity-Only Live Trader"
                    )
                return

            # DRY RUN: Show what would happen
            if self.dry_run:
                print("\n" + "=" * 70)
                print("DRY RUN MODE - NO TRADES EXECUTED")
                print("=" * 70)
                print("\nConfirmed positions that WOULD be traded (equity-only):")
                for ticker, weight in sorted(
                    confirmed_weights.items(), key=lambda x: x[1], reverse=True
                ):
                    print(f"  {ticker}: {weight*100:.2f}%")
                print("\nRejected positions:")
                for ticker in rejected.keys():
                    print(f"  {ticker}")
                print("\nRe-run with --live flag to execute trades")
                return

            # STEP 3: Execute trades via IB
            print("\n" + "=" * 70)
            print("STEP 3: EXECUTION VIA INTERACTIVE BROKERS (EQUITY-ONLY)")
            print("=" * 70)

            atr_data = signals.get("atr_data", {})
            atr_data = {ticker: atr for ticker, atr in atr_data.items() if ticker in confirmed_weights}

            print(f"\n📐 ATR Data for stops: {len(atr_data)} tickers")
            if atr_data:
                for ticker, atr in list(atr_data.items())[:3]:
                    print(f"  {ticker}: ${atr:.2f}")
                if len(atr_data) > 3:
                    print(f"  ... and {len(atr_data) - 3} more")
            else:
                print("  ⚠️ WARNING: No ATR data available - stops will NOT be placed!")

            with IBExecutor(port=self.ib_port, account=self.ib_account) as ib_exec:
                if not ib_exec.connected:
                    print("\nERROR: Failed to connect to IB Gateway")
                    print("Make sure IB Gateway is running on port", self.ib_port)
                    if self.telegram:
                        self.telegram.send_error_alert(
                            "Failed to connect to IB Gateway", "Equity-Only Live Trader"
                        )
                    return

                print("+ Connected to IB Gateway")

                account_value = ib_exec.get_account_value()
                print(f"+ Account value: ${account_value:,.2f}")

                if account_value <= 0:
                    print("\nERROR: Invalid account value")
                    return

                position_manager = PositionManager(ib_exec.ib, account=self.ib_account)

                positions_before = ib_exec.get_current_positions()
                print(f"+ Current positions: {len(positions_before)}")

                trades = ib_exec.execute_rebalance(
                    confirmed_weights,
                    position_manager=position_manager,
                    atr_data=atr_data,
                )

                positions_after = ib_exec.get_current_positions()

                # ENFORCE TOP 10: Close any positions not in confirmed_weights
                positions_to_close = [
                    ticker for ticker in positions_after.keys() if ticker not in confirmed_weights
                ]
                if positions_to_close:
                    print(
                        f"\n⚠️ Found {len(positions_to_close)} positions not in top 10 - closing..."
                    )
                    for ticker in positions_to_close:
                        try:
                            contract = ib_exec.create_cfd_contract(ticker)
                            if contract:
                                quantity = int(abs(positions_after[ticker]))
                                if quantity > 0:
                                    action = "SELL" if positions_after[ticker] > 0 else "BUY"
                                    print(f"  Closing {ticker} ({quantity} shares)...")
                                    trade = ib_exec.place_order(contract, quantity, action)
                                    if trade:
                                        print(f"    ✓ Closed {ticker}")
                        except Exception as e:
                            print(f"    ✗ Error closing {ticker}: {e}")

                    positions_after = ib_exec.get_current_positions()

                print("\n" + "=" * 70)
                print("EXECUTION COMPLETE (EQUITY-ONLY)")
                print("=" * 70)
                print(f"Confirmed entries: {len(confirmed_weights)}")
                print(f"Rejected entries: {len(rejected)}")
                print(f"Trades executed: {len(trades) if trades else 0}")
                print(f"Final positions: {len(positions_after)}")
                if len(positions_after) > len(confirmed_weights):
                    print(
                        f"⚠️ WARNING: {len(positions_after)} positions but only {len(confirmed_weights)} confirmed!"
                    )
                print("=" * 70)

                # Telegram notification could be wired similarly to live_trader_simple if desired.

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback

            traceback.print_exc()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Equity-only live trader for Macro Quadrant strategy"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4001,
        help="IB port (4001=live Gateway, 4002=paper Gateway)",
    )
    parser.add_argument(
        "--account",
        type=str,
        default="",
        help="IBKR account ID to use (overrides IB_ACCOUNT env).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute live trades (default is dry run)",
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Disable Telegram notifications",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    trader = EquityOnlyLiveTrader(
        ib_port=args.port,
        dry_run=not args.live,
        enable_telegram=not args.no_telegram,
        ib_account=args.account,
    )
    trader.run()


if __name__ == "__main__":
    main()

