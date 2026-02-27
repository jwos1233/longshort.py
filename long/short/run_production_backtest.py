"""
Run Production v3.0 Backtest - Top 10 + ATR 2.0x

This is the long/short project bootstrap copy.
It runs using the existing Macro_Quadrant_Strategy modules from repo root.

Usage:
  python run_production_backtest.py
  python run_production_backtest.py --start 2025-01-01 --end 2025-11-30
  python run_production_backtest.py --start 2025-01-01 --end 2025-11-30 --report report.json
"""

import argparse
from datetime import datetime, timedelta, date

from _bootstrap_path import ensure_repo_root_on_path

ensure_repo_root_on_path()

from quad_portfolio_backtest import QuadrantPortfolioBacktest  # noqa: E402


# Setup
INITIAL_CAPITAL = 50000
BACKTEST_YEARS = 5

# Use yesterday's date to ensure finalized close prices (matches signal_generator.py)
today = date.today()
default_end = datetime.combine(today - timedelta(days=1), datetime.min.time())
default_start = default_end - timedelta(days=BACKTEST_YEARS * 365 + 100)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Macro Quadrant backtest (long/short bootstrap)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--report",
        "-o",
        type=str,
        default=None,
        help="Output path for LLM JSON report (always generated; this overrides default filename)",
    )
    parser.add_argument("--no-chart", action="store_true", help="Skip chart display")
    return parser.parse_args()


def main():
    args = parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else default_start
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else default_end

    print("=" * 70)
    print("PRODUCTION v3.0: TOP 10 + ATR 2.0x STOP LOSS + EMA SMOOTHING (bootstrap)")
    print("=" * 70)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Momentum Lookback: 20 days")
    print(f"EMA Smoothing: 20-period (quad scores)")
    print(f"Max Positions: 10")
    print(f"Stop Loss: 2.0x ATR (14-day)")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("=" * 70)
    print()

    backtest = QuadrantPortfolioBacktest(
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

    # Print summary
    print("\n" + "=" * 70)
    print("PRODUCTION v3.0 PERFORMANCE")
    print("=" * 70)
    print(f"Total Return:      {results['total_return']:.2f}%")
    print(f"Annualized Return: {results['annual_return']:.2f}%")
    print(f"Sharpe Ratio:      {results['sharpe']:.2f}")
    print(f"Max Drawdown:      {results['max_drawdown']:.2f}%")
    print(f"Volatility:        {results['annual_vol']:.2f}%")
    print(f"Final Value:       ${results['final_value']:,.2f}")
    print("=" * 70)

    # Generate LLM report (JSON with P/L attribution, dominant quads, ticker breakdown)
    report = backtest.generate_llm_report(
        output_path=args.report,
        report_start_date=start_date,
        report_end_date=end_date,
    )
    if report["meta"].get("ytd_return_pct") is not None:
        print(f"YTD Return:        {report['meta']['ytd_return_pct']:.2f}%")

    backtest.print_spy_comparison()
    backtest.print_annual_breakdown()

    if not args.no_chart:
        print("\nGenerating P/L Chart...")
        backtest.plot_results()
        print("\nChart displayed! Close the chart window to continue.")
    print("=" * 70)


if __name__ == "__main__":
    main()

