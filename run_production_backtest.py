"""
Run Production v3.0 Backtest - Top 10 + ATR 2.0x
"""

from quad_portfolio_backtest import QuadrantPortfolioBacktest
from datetime import datetime, timedelta

# Setup
INITIAL_CAPITAL = 50000
BACKTEST_YEARS = 5

# Use yesterday's date to ensure finalized close prices (matches signal_generator.py)
from datetime import date
today = date.today()
end_date = datetime.combine(today - timedelta(days=1), datetime.min.time())
start_date = end_date - timedelta(days=BACKTEST_YEARS * 365 + 100)

print("="*70)
print("PRODUCTION v3.0: TOP 10 + ATR 2.0x STOP LOSS + EMA SMOOTHING")
print("="*70)
print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
print(f"Momentum Lookback: 20 days")
print(f"EMA Smoothing: 20-period (quad scores)")
print(f"Max Positions: 10")
print(f"Stop Loss: 2.0x ATR (14-day)")
print(f"Period: ~{BACKTEST_YEARS} years")
print("="*70)
print()

# Run production strategy (with EMA smoothing enabled)
backtest = QuadrantPortfolioBacktest(
    start_date=start_date,
    end_date=end_date,
    initial_capital=INITIAL_CAPITAL,
    momentum_days=20,
    max_positions=10,
    atr_stop_loss=2.0,
    atr_period=14,
    ema_smoothing_period=20  # EMA smoothing for quad scores
)

results = backtest.run_backtest()

# Print summary
print("\n" + "="*70)
print("PRODUCTION v3.0 PERFORMANCE")
print("="*70)
print(f"Total Return:      {results['total_return']:.2f}%")
print(f"Annualized Return: {results['annual_return']:.2f}%")
print(f"Sharpe Ratio:      {results['sharpe']:.2f}")
print(f"Max Drawdown:      {results['max_drawdown']:.2f}%")
print(f"Volatility:        {results['annual_vol']:.2f}%")
print(f"Final Value:       ${results['final_value']:,.2f}")

print("="*70)

# Show comparison to buy & hold SPY
backtest.print_spy_comparison()

# Annual breakdown
backtest.print_annual_breakdown()

# Plot results
print("\nGenerating P/L Chart...")
backtest.plot_results()

print("\n" + "="*70)
print("Chart displayed! Close the chart window to continue.")
print("="*70)



