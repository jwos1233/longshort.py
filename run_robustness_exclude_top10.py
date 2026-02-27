"""
Run Robustness Backtest: Equity-Only MINUS Top 10 P/L Tickers
==============================================================

Same logic as run_equity_only_backtest.py, but EXCLUDES the top 10
P/L contributors to test robustness. Are returns diversified or
concentrated in a few names?

Usage:
  python run_robustness_exclude_top10.py
  python run_robustness_exclude_top10.py --exclude "BMNR,SGML,LAC,UUUU,RGTI,LTBR,ALB,CRWV,OKLO,MP"
  (In PowerShell/CMD you MUST use quotes around the comma-separated list)
"""

import argparse
from datetime import datetime, timedelta, date

from robustness_backtest import ExcludeTopRobustnessBacktest, EXCLUDED_TOP10_DEFAULT


INITIAL_CAPITAL = 50000
BACKTEST_MONTHS = 18


today = date.today()
default_end = datetime.combine(today - timedelta(days=1), datetime.min.time())
default_start = default_end - timedelta(days=int(365 * BACKTEST_MONTHS / 12))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run robustness backtest (equity-only, top 10 P/L excluded)"
    )
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=EXCLUDED_TOP10_DEFAULT,
        metavar="TICKER",
        help="Tickers to exclude (space-separated). Omit to use default top 10.",
    )
    parser.add_argument(
        "--report",
        "-o",
        type=str,
        default=None,
        help="Output path for LLM JSON report",
    )
    parser.add_argument("--no-chart", action="store_true", help="Skip chart display")
    return parser.parse_args()


def main():
    args = parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else default_start
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else default_end

    if start_date > end_date:
        start_date, end_date = end_date, start_date
        print("(Start was after end; swapped to valid range.)")

    exclude_list = list(args.exclude) if args.exclude else EXCLUDED_TOP10_DEFAULT
    exclude_list = [str(t).strip().upper() for t in exclude_list if t]

    print("=" * 70)
    print("ROBUSTNESS: EQUITY-ONLY MINUS TOP 10 P/L TICKERS")
    print("=" * 70)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Excluded tickers: {exclude_list}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("=" * 70)
    print()

    backtest = ExcludeTopRobustnessBacktest(
        exclude_tickers=exclude_list,
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
    print("ROBUSTNESS PERFORMANCE (UNHEDGED - TOP 10 EXCLUDED)")
    print("=" * 70)
    print(f"Total Return:      {results['total_return']:.2f}%")
    print(f"Annualized Return: {results['annual_return']:.2f}%")
    print(f"Sharpe Ratio:      {results['sharpe']:.2f}")
    print(f"Max Drawdown:      {results['max_drawdown']:.2f}%")
    print(f"Volatility:        {results['annual_vol']:.2f}%")
    print(f"Final Value:       ${results['final_value']:,.2f}")
    print("=" * 70)

    if "hedged" in results:
        h = results["hedged"]
        print("\n" + "=" * 70)
        print("ROBUSTNESS + SHORT HEDGES (HEDGED)")
        print("=" * 70)
        print(f"Total Return:      {h['total_return']:.2f}%")
        print(f"Annualized Return: {h['annual_return']:.2f}%")
        print(f"Sharpe Ratio:      {h['sharpe']:.2f}")
        print(f"Max Drawdown:      {h['max_drawdown']:.2f}%")
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
        backtest.print_annual_breakdown(
            pv=backtest.hedged_portfolio_value, label="Robustness (Hedged)"
        )

    if not args.no_chart:
        print("\nGenerating P/L Chart...")
        backtest.plot_results()
        print("\nChart displayed! Close the chart window to continue.")
    print("=" * 70)


if __name__ == "__main__":
    main()
