# Macro Quadrant Rotation Strategy

A systematic trading strategy that rotates between economic regimes (quadrants) based on growth and inflation dynamics, executed via Interactive Brokers.

## Strategy Overview

The strategy divides market conditions into 4 quadrants based on growth and inflation:

| Quadrant | Growth | Inflation | Description |
|----------|--------|-----------|-------------|
| **Q1** | ↑ Rising | ↓ Falling | Goldilocks (Tech, Growth) |
| **Q2** | ↑ Rising | ↑ Rising | Reflation (Commodities, Energy) |
| **Q3** | ↓ Falling | ↑ Rising | Stagflation (Gold, Defensive) |
| **Q4** | ↓ Falling | ↓ Falling | Deflation (Bonds, Cash) |

### How It Works

1. **Quadrant Detection**: Uses 20-day momentum of key indicators (QQQ, XLE, GLD, TLT, etc.) to score each quadrant
2. **EMA Smoothing**: Applies 20-period EMA to quad scores to reduce whipsaws and noise
3. **Top 2 Selection**: Selects the top 2 performing quadrants each day
4. **Volatility Chasing**: Within each quadrant, allocates more to volatile assets (volatility chasing)
5. **EMA Filter**: Only enters positions above their 50-day EMA (trend filter)
6. **Position Limits**: Maximum 10 positions, top-weighted by volatility
7. **Stop Losses**: 2.0x ATR (14-day) stop loss on all positions

### Key Features

- **1-Day Entry Lag**: Uses yesterday's quadrant signals to prevent forward-looking bias
- **Live EMA Confirmation**: Confirms entries using today's EMA status (responsive to current market)
- **Realistic Execution**: Enters at next-day open prices (no look-ahead bias)
- **ATR Stop Losses**: Dynamic stops based on volatility (2.0x ATR)

## Execution Platform

**Interactive Brokers (IB)** via IB Gateway/IBKR API
- Uses CFDs for execution
- Supports both live and paper trading
- Port 4001 = Live Gateway
- Port 4002 = Paper Gateway

## Files

### Core Execution Scripts

- **`live_trader_simple.py`** - Live trading execution
  - Generates signals from yesterday's close
  - Confirms with today's EMA
  - Executes trades via IB
  
- **`run_production_backtest.py`** - Production backtest
  - Backtests strategy over 5 years
  - Includes EMA smoothing, volatility chasing, ATR stops
  
- **`hypothetical_signals.py`** - Signal generation only
  - Generates theoretical signals
  - Sends to Telegram (no execution)

### Core Modules

- **`signal_generator.py`** - Generates trading signals
- **`quad_portfolio_backtest.py`** - Backtest engine
- **`ib_executor.py`** - IB execution interface
- **`position_manager.py`** - Position and stop loss management
- **`telegram_notifier.py`** - Telegram notifications
- **`config.py`** - Strategy configuration
- **`strategy_config.py`** - Execution settings

## Usage

### Live Trading

```bash
# Dry run (see what would happen)
python live_trader_simple.py --port 4001

# Live execution
python live_trader_simple.py --port 4001 --live

# Disable Telegram
python live_trader_simple.py --port 4001 --live --no-telegram
```

### Production Backtest

```bash
python run_production_backtest.py
```

### Hypothetical Signals

```bash
python hypothetical_signals.py
```

## Requirements

See `requirements.txt` for full list. Key dependencies:
- `ib_insync` - Interactive Brokers API
- `yfinance` - Market data
- `pandas`, `numpy` - Data processing
- `python-telegram-bot` - Telegram notifications

## Configuration

Edit `config.py` for:
- Quadrant allocations (`QUAD_ALLOCATIONS`)
- Quadrant indicators (`QUAD_INDICATORS`)
- Telegram credentials (`TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`)
- Crypto proxy execution (`BTC_PROXY_BASKET`, `BTC_PROXY_MAX_POSITIONS`)

### Crypto proxy execution (BTC-USD → proxies)

If your broker cannot trade spot crypto tickers like `BTC-USD`, the strategy can keep `BTC-USD` in the model universe but **execute that crypto sleeve using equity proxies**.

- **Where it applies**: When the strategy would allocate to `BTC-USD` (most commonly in Q1), that weight is redistributed into the proxy universe.
- **Selection**: Proxies must pass the same **50-day EMA filter** (only above EMA).
- **Weighting inside the crypto sleeve**: **Volatility chasing** (higher volatility gets higher weight), capped to **`BTC_PROXY_MAX_POSITIONS` (default 10)**.
- **Note**: If none of the proxy tickers have data or pass the EMA filter, the crypto sleeve becomes cash for that run.

Edit `strategy_config.py` for:
- Discretionary positions to ignore
- Contract types to manage

## Performance

**Backtest Results (5 years, with EMA smoothing):**
- Total Return: ~10,600%
- Annualized Return: ~94%
- Sharpe Ratio: ~2.65
- Max Drawdown: ~-23%

*Past performance does not guarantee future results*

## Notes

- Strategy uses **yesterday's close prices** for signal generation (finalized data)
- Entry confirmation uses **today's EMA** (live/current data)
- All positions entered at **next-day open** (realistic execution)
- Stop losses are placed automatically via IB API

