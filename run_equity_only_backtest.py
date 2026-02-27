"""
Run Equity-Only Backtest
========================

This variant:
  - Uses the same macro quadrant logic and production settings as
    run_production_backtest.py.
  - BUT only takes risk in *equity* ETFs / stocks.
  - Any allocation to NON_EQUITY_ETFS (bonds, commodities, crypto, vol, etc.)
    is treated as cash (no trade).

Usage:
  python run_equity_only_backtest.py
  python run_equity_only_backtest.py --start 2023-01-01 --end 2024-12-31
  python run_equity_only_backtest.py --start 2023-01-01 --end 2024-12-31 --report eq_only_report.json
"""

import argparse
from datetime import datetime, timedelta, date

from equity_only_backtest import EquityOnlyBacktest


INITIAL_CAPITAL = 50000
# Default window: last 18 months, same as production runner.
BACKTEST_MONTHS = 18


today = date.today()
default_end = datetime.combine(today - timedelta(days=1), datetime.min.time())
default_start = default_end - timedelta(days=int(365 * BACKTEST_MONTHS / 12))


def parse_args():
    parser = argparse.ArgumentParser(description="Run Macro Quadrant equity-only backtest")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--report",
        "-o",
        type=str,
        default=None,
        help="Output path for LLM JSON report (overrides default filename)",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip chart display",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else default_start
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else default_end

    if start_date > end_date:
        start_date, end_date = end_date, start_date
        print("(Start was after end; swapped to use valid range.)")

    print("=" * 70)
    print("EQUITY-ONLY BACKTEST: TOP 10 + ATR 2.0x STOP + EMA SMOOTHING")
    print("=" * 70)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Momentum Lookback: 20 days")
    print(f"EMA Smoothing: 20-period (quad scores)")
    print(f"Max Positions: 10")
    print(f"Stop Loss: 2.0x ATR (14-day)")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("Non-equity sleeves (bonds, commodities, crypto, vol) = CASH (no trade)")
    print("=" * 70)
    print()

    backtest = EquityOnlyBacktest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=INITIAL_CAPITAL,
        momentum_days=20,
        max_positions=10,
        atr_stop_loss=2.0,
        atr_period=14,
        ema_smoothing_period=20,
    )

    results = backtest.run_backtest()

    print("\n" + "=" * 70)
    print("EQUITY-ONLY PERFORMANCE (UNHEDGED)")
    print("=" * 70)
    print(f"Total Return:      {results['total_return']:.2f}%")
    print(f"Annualized Return: {results['annual_return']:.2f}%")
    print(f"Sharpe Ratio:      {results['sharpe']:.2f}")
    print(f"Max Drawdown:      {results['max_drawdown']:.2f}%")
    print(f"Volatility:        {results['annual_vol']:.2f}%")
    print(f"Final Value:       ${results['final_value']:,.2f}")
    print("=" * 70)

    # Hedged curve, if available (short legs are still allowed; they use SPY / equity proxies)
    if 'hedged' in results:
        h = results['hedged']
        print("\n" + "=" * 70)
        print("EQUITY-ONLY + SHORT HEDGES (HEDGED CURVE)")
        print("=" * 70)
        print(f"Total Return:      {h['total_return']:.2f}%")
        print(f"Annualized Return: {h['annual_return']:.2f}%")
        print(f"Sharpe Ratio:      {h['sharpe']:.2f}")
        print(f"Max Drawdown:      {h['max_drawdown']:.2f}%")
        print(f"Volatility:        {h['annual_vol']:.2f}%")
        print(f"Final Value:       ${h['final_value']:,.2f}")
        print("=" * 70)

    report = backtest.generate_llm_report(
        output_path=args.report,
        report_start_date=start_date,
        report_end_date=end_date,
    )
    if report["meta"].get("ytd_return_pct") is not None:
        print(f"YTD Return:        {report['meta']['ytd_return_pct']:.2f}%")

    backtest.print_spy_comparison()
    backtest.print_annual_breakdown()
    backtest.print_ticker_attribution(top_n=30)
    if getattr(backtest, "hedged_portfolio_value", None) is not None:
        backtest.print_annual_breakdown(pv=backtest.hedged_portfolio_value, label="Equity-Only (Hedged)")

    if not args.no_chart:
        print("\nGenerating P/L Chart...")
        backtest.plot_results()
        print("\nChart displayed! Close the chart window to continue.")
    print("=" * 70)


if __name__ == "__main__":
    main()

