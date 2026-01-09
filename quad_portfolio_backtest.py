"""
Macro Quadrant Portfolio Backtest - PRODUCTION VERSION
===========================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from config import QUAD_ALLOCATIONS, QUADRANT_DESCRIPTIONS

# Quadrant indicators for scoring (same as signal_generator.py)
QUAD_INDICATORS = {
    'Q1': ['QQQ', 'VUG', 'IWM', 'BTC-USD'],
    'Q2': ['XLE', 'DBC'],
    'Q3': ['GLD', 'LIT'],
    'Q4': ['TLT', 'XLU', 'VIXY']
}

# Backtest leverage controls
BASE_QUAD_LEVERAGE = 1.5       # 1.5x exposure for all quads
Q1_LEVERAGE_MULTIPLIER = 1.0   # Q1 gets same as base (1.5x) - no extra boost

# Manual overrides for assets that must be fetched even if not in current
# allocation map (keeps backtests aligned with latest production universe).
ADDITIONAL_BACKTEST_TICKERS = ['LIT', 'AA', 'PALL', 'VALT']

class QuadrantPortfolioBacktest:
    def __init__(self, start_date, end_date, initial_capital=50000, 
                 momentum_days=50, ema_period=50, vol_lookback=30, max_positions=None,
                 atr_stop_loss=None, atr_period=14, ema_smoothing_period=20):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.momentum_days = momentum_days
        self.ema_period = ema_period
        self.vol_lookback = vol_lookback
        self.max_positions = max_positions  # If set, only trade top N positions
        self.atr_stop_loss = atr_stop_loss  # ATR multiplier for stop loss (None = no stops)
        self.atr_period = atr_period  # ATR lookback period (default 14)
        self.ema_smoothing_period = ema_smoothing_period  # EMA smoothing for quad scores (default 20)
        
        self.price_data = None
        self.open_data = None
        self.atr_data = None
        self.ema_data = None
        self.volatility_data = None
        self.portfolio_value = None
        self.quad_history = None
    
    def fetch_data(self):
        """Download price data for all tickers (Close for signals, Open for execution)"""
        all_tickers = []
        for quad_assets in QUAD_ALLOCATIONS.values():
            all_tickers.extend(quad_assets.keys())
        all_tickers.extend(ADDITIONAL_BACKTEST_TICKERS)
        all_tickers = sorted(set(all_tickers))
        
        print(f"Fetching data for {len(all_tickers)} tickers...")
        
        # Use yesterday's date to ensure finalized close prices (no look-ahead bias)
        # This matches signal_generator.py behavior
        from datetime import date
        today = date.today()
        end_date_actual = datetime.combine(today - timedelta(days=1), datetime.min.time())
        
        # Add buffer for momentum calculation
        buffer_days = max(self.momentum_days, self.ema_period, self.vol_lookback) + 10
        fetch_start = pd.to_datetime(self.start_date) - timedelta(days=buffer_days)
        
        print(f"Period: {fetch_start.date()} to {end_date_actual.date()} (using finalized close prices)")
        
        price_data = {}
        open_data = {}
        for ticker in all_tickers:
            try:
                # Use start/end dates to fetch full historical range
                # Add 1 day to end_date_actual to ensure we get data up to that date
                end_date_for_download = end_date_actual + timedelta(days=1)
                data = yf.download(ticker, start=fetch_start, end=end_date_for_download, 
                                 progress=False, auto_adjust=True)
                # Filter to our desired date range (inclusive of end_date_actual)
                if len(data) > 0:
                    data = data[data.index.date <= end_date_actual.date()]
                
                # Extract Close prices (for signals, momentum, EMA)
                if isinstance(data.columns, pd.MultiIndex):
                    if 'Close' in data.columns.get_level_values(0):
                        prices = data['Close']
                    if 'Open' in data.columns.get_level_values(0):
                        opens = data['Open']
                else:
                    if 'Close' in data.columns:
                        prices = data['Close']
                    else:
                        continue
                    if 'Open' in data.columns:
                        opens = data['Open']
                    else:
                        continue
                
                if isinstance(prices, pd.DataFrame):
                    prices = prices.iloc[:, 0]
                if isinstance(opens, pd.DataFrame):
                    opens = opens.iloc[:, 0]
                
                if len(prices) > 100 and len(opens) > 100:
                    price_data[ticker] = prices
                    open_data[ticker] = opens
                    print(f"+ {ticker}: {len(prices)} days")
                    
            except Exception as e:
                print(f"- {ticker}: {e}")
                continue
        
        self.price_data = pd.DataFrame(price_data)
        self.price_data = self.price_data.ffill().bfill()
        
        self.open_data = pd.DataFrame(open_data)
        self.open_data = self.open_data.ffill().bfill()
        
        print(f"\nLoaded {len(self.price_data.columns)} tickers, {len(self.price_data)} days")
        print(f"  Close prices: for signals/momentum/EMA")
        print(f"  Open prices: for realistic execution (next-day open)")
        
        # Calculate 50-day EMA
        print(f"Calculating {self.ema_period}-day EMA for trend filter...")
        self.ema_data = self.price_data.ewm(span=self.ema_period, adjust=False).mean()
        
        # Calculate volatility (rolling std of returns)
        print(f"Calculating {self.vol_lookback}-day rolling volatility for volatility chasing...")
        returns = self.price_data.pct_change()
        self.volatility_data = returns.rolling(window=self.vol_lookback).std() * np.sqrt(252)
        
        # Calculate ATR if stop loss is enabled
        if self.atr_stop_loss is not None:
            print(f"Calculating {self.atr_period}-day ATR for stop loss (multiplier: {self.atr_stop_loss}x)...")
            # Simplified ATR using daily returns volatility
            daily_returns = self.price_data.pct_change().abs()
            self.atr_data = daily_returns.rolling(window=self.atr_period).mean() * self.price_data
    
    def calculate_quad_scores(self):
        """
        Calculate momentum scores for each quadrant using QUAD_INDICATORS
        (same as signal_generator.py - uses indicators for scoring, not all assets)
        """
        print(f"\nCalculating {self.momentum_days}-day momentum scores (using QUAD_INDICATORS)...")
        
        # Calculate momentum for all assets
        momentum = self.price_data.pct_change(self.momentum_days)
        
        # Score each quadrant using QUAD_INDICATORS (same as signal generator)
        quad_scores = pd.DataFrame(index=momentum.index)
        
        for quad, indicators in QUAD_INDICATORS.items():
            # For each date, calculate quad score the same way signal_generator does
            quad_score_series = pd.Series(index=momentum.index, dtype=float)
            
            for date in momentum.index:
                quad_scores_list = []
                for ticker in indicators:
                    if ticker in momentum.columns:
                        # Get momentum for this ticker on this date
                        ticker_momentum = momentum.loc[date, ticker]
                        if pd.notna(ticker_momentum):
                            quad_scores_list.append(ticker_momentum)
                
                # Average across indicators for this quadrant (same as signal_generator)
                if quad_scores_list:
                    quad_score_series.loc[date] = np.mean(quad_scores_list)
                else:
                    quad_score_series.loc[date] = 0.0
            
            quad_scores[quad] = quad_score_series
        
        return quad_scores
    
    def determine_top_quads(self, quad_scores):
        """Determine top 2 quadrants for each day"""
        top_quads = pd.DataFrame(index=quad_scores.index)
        
        for date in quad_scores.index:
            scores = quad_scores.loc[date].sort_values(ascending=False)
            top_quads.loc[date, 'Top1'] = scores.index[0]
            top_quads.loc[date, 'Top2'] = scores.index[1]
            top_quads.loc[date, 'Score1'] = scores.iloc[0]
            top_quads.loc[date, 'Score2'] = scores.iloc[1]
        
        return top_quads
    
    def calculate_target_weights(self, top_quads):
        """Calculate target portfolio weights with volatility chasing"""
        weights = pd.DataFrame(0.0, index=top_quads.index, 
                              columns=self.price_data.columns)
        
        for date in top_quads.index:
            top1 = top_quads.loc[date, 'Top1']
            top2 = top_quads.loc[date, 'Top2']
            score1 = top_quads.loc[date, 'Score1']
            score2 = top_quads.loc[date, 'Score2']
            
            # Process each quad separately with volatility weighting
            final_weights = {}
            
            # UNIFORM LEVERAGE: 1.5x base exposure for all quads
            for quad in (top1, top2):
                quad_weight = BASE_QUAD_LEVERAGE
                if quad == 'Q1':
                    quad_weight *= Q1_LEVERAGE_MULTIPLIER
                    
                # Get tickers for this quad
                quad_tickers = [t for t in QUAD_ALLOCATIONS[quad].keys() 
                              if t in self.price_data.columns]
                
                if not quad_tickers:
                    continue
                
                # Get volatilities for this date
                quad_vols = {}
                for ticker in quad_tickers:
                    if ticker in self.volatility_data.columns and date in self.volatility_data.index:
                        vol = self.volatility_data.loc[date, ticker]
                        if pd.notna(vol) and vol > 0:
                            quad_vols[ticker] = vol
                
                if not quad_vols:
                    continue
                
                # Calculate DIRECT volatility weights (higher vol = higher weight / volatility chasing)
                direct_vols = {t: v for t, v in quad_vols.items()}
                total_vol = sum(direct_vols.values())
                
                # Normalize to quad_weight (Q1=3x, others=2x)
                vol_weights = {t: (v / total_vol) * quad_weight 
                             for t, v in direct_vols.items()}
                
                # Apply EMA filter - assets below EMA get zero weight (held as cash)
                for ticker, weight in vol_weights.items():
                    if ticker in self.ema_data.columns and date in self.price_data.index:
                        price = self.price_data.loc[date, ticker]
                        ema = self.ema_data.loc[date, ticker]
                        
                        if pd.notna(price) and pd.notna(ema) and price > ema:
                            # Pass EMA filter: add to final weights
                            if ticker in final_weights:
                                final_weights[ticker] += weight
                            else:
                                final_weights[ticker] = weight
            
            # Filter to top N positions if max_positions is set
            if self.max_positions and len(final_weights) > self.max_positions:
                # Sort by weight and keep top N
                sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
                top_n_weights = dict(sorted_weights[:self.max_positions])
                
                # Re-normalize to maintain total leverage
                original_total = sum(final_weights.values())
                new_total = sum(top_n_weights.values())
                scale_factor = original_total / new_total if new_total > 0 else 1
                
                final_weights = {t: w * scale_factor for t, w in top_n_weights.items()}
            
            # Apply final weights to the weights DataFrame
            for ticker, weight in final_weights.items():
                weights.loc[date, ticker] = weight
        
        return weights
    
    def run_backtest(self):
        """Run the complete backtest with TRUE 1-day entry confirmation"""
        print("=" * 70)
        print("QUADRANT PORTFOLIO BACKTEST - PRODUCTION VERSION")
        print("=" * 70)
        
        # Fetch data
        self.fetch_data()
        
        # Calculate quadrant scores
        quad_scores = self.calculate_quad_scores()
        
        # Warmup period
        warmup = self.momentum_days
        quad_scores.iloc[:warmup] = np.nan
        
        # Apply EMA smoothing to quad scores if enabled
        if self.ema_smoothing_period and self.ema_smoothing_period > 0:
            print(f"\nApplying {self.ema_smoothing_period}-period EMA smoothing to quad scores...")
            smoothed_scores = pd.DataFrame(index=quad_scores.index, columns=quad_scores.columns)
            for quad in quad_scores.columns:
                smoothed_scores[quad] = quad_scores[quad].ewm(
                    span=self.ema_smoothing_period, 
                    adjust=False
                ).mean()
            quad_scores = smoothed_scores
            print(f"✓ Smoothed scores calculated")
        
        # Determine top 2 quads each day
        print("\nDetermining top 2 quadrants daily...")
        top_quads = self.determine_top_quads(quad_scores.iloc[warmup:])
        self.quad_history = top_quads
        
        # Calculate target weights
        print("Calculating target portfolio weights...")
        target_weights = self.calculate_target_weights(top_quads)
        self.target_weights = target_weights  # Store for access
        
        # Simulate portfolio with EVENT-DRIVEN rebalancing + TRUE 1-DAY ENTRY LAG
        print("Simulating portfolio with TRUE 1-day entry confirmation + REALISTIC EXECUTION...")
        print("  Macro signals: T-1 lag (trade yesterday's regime)")
        print("  Entry confirmation: Check TODAY's EMA (live/current)")
        print("  Execution timing: NEXT DAY OPEN (realistic fill)")
        print("  Exit rule: Immediate (no lag)")
        print("  P&L: Overnight at OLD positions, Intraday at NEW positions")
        
        portfolio_value = pd.Series(self.initial_capital, index=target_weights.index)
        actual_positions = pd.Series(0.0, index=target_weights.columns)  # Current holdings
        prev_positions = pd.Series(0.0, index=target_weights.columns)  # Track previous positions for cost calculation
        pending_entries = {}  # {ticker: target_weight} - waiting for confirmation
        entry_prices = {}  # {ticker: entry_price} - for stop loss calculation
        entry_dates = {}  # {ticker: entry_date} - for tracking entry history
        entry_atrs = {}  # {ticker: atr_at_entry} - for stop calculation
        
        prev_top_quads = None
        prev_ema_status = {}
        rebalance_count = 0
        entries_confirmed = 0
        entries_rejected = 0
        trades_skipped = 0  # Track trades skipped due to minimum threshold
        stops_hit = 0  # Track stop losses
        total_costs = 0.0  # Track cumulative trading costs
        
        
        # Trading cost per leg (10 basis points = 0.10%)
        COST_PER_LEG_BPS = 10  # 10 basis points = 0.0010
        
        # Minimum trade size threshold (only trade if delta > this %)
        MIN_TRADE_THRESHOLD = 0.05  # 5% minimum trade size
        
        for i in range(1, len(target_weights)):
            date = target_weights.index[i]
            prev_date = target_weights.index[i-1]
            
            # ===== CRITICAL: LAG STRUCTURE TO PREVENT FORWARD-LOOKING BIAS =====
            # MACRO SIGNALS (Quad Rankings): T-1 lag
            #   - On Day T, we trade based on Day T-1's quad rankings
            #   - This prevents forward-looking bias in regime detection
            # 
            # ENTRY CONFIRMATION (EMA Filter): T+0 (current/live)
            #   - We check TODAY's EMA to confirm entry (not yesterday's)
            #   - This is the key difference: responsive to current market
            # ===================================================================
            
            if i >= 1:
                target_date = target_weights.index[i-1]  # YESTERDAY (T-1 for quad signals)
                current_top_quads = (top_quads.loc[target_date, 'Top1'], 
                                   top_quads.loc[target_date, 'Top2'])
                
                # Check EMA status for YESTERDAY (for change detection)
                yesterday_ema_status = {}
                for ticker in target_weights.columns:
                    if ticker in self.ema_data.columns and target_date in self.price_data.index:
                        price = self.price_data.loc[target_date, ticker]
                        ema = self.ema_data.loc[target_date, ticker]
                        if pd.notna(price) and pd.notna(ema):
                            yesterday_ema_status[ticker] = price > ema
                
                # Check EMA status for TODAY (for entry confirmation) - THIS IS THE KEY DIFFERENCE!
                today_ema_status = {}
                for ticker in target_weights.columns:
                    if ticker in self.ema_data.columns and date in self.price_data.index:
                        price = self.price_data.loc[date, ticker]
                        ema = self.ema_data.loc[date, ticker]
                        if pd.notna(price) and pd.notna(ema):
                            today_ema_status[ticker] = price > ema
                
                # Get current target weights (based on yesterday's signals)
                current_targets = target_weights.loc[target_date]
                
                # Process pending entries - confirm if still above EMA TODAY
                confirmed_entries = {}
                for ticker, weight in list(pending_entries.items()):
                    # Check if STILL above EMA using TODAY's data
                    if ticker in today_ema_status and today_ema_status[ticker]:
                        # Confirmed! Enter the position
                        confirmed_entries[ticker] = weight
                        entries_confirmed += 1
                    else:
                        # Rejected - dropped below EMA
                        entries_rejected += 1
                    # Remove from pending regardless
                    del pending_entries[ticker]
                
                # Check ATR stop losses (if enabled)
                stop_loss_exits = []
                if self.atr_stop_loss is not None and date in self.price_data.index:
                    for ticker in actual_positions[actual_positions > 0].index:
                        if ticker in entry_prices and ticker in self.atr_data.columns:
                            current_price = self.price_data.loc[date, ticker]
                            entry_price = entry_prices[ticker]
                            atr = self.atr_data.loc[date, ticker]
                            
                            if pd.notna(current_price) and pd.notna(atr) and pd.notna(entry_price):
                                stop_price = entry_price - (atr * self.atr_stop_loss)
                                
                                # Check if stop hit
                                if current_price <= stop_price:
                                    stop_loss_exits.append(ticker)
                                    actual_positions[ticker] = 0.0
                                    del entry_prices[ticker]
                                    if ticker in entry_dates:
                                        del entry_dates[ticker]
                                    if ticker in entry_atrs:
                                        del entry_atrs[ticker]
                                    stops_hit += 1
                
                # Determine if we need to rebalance
                should_rebalance = False
                
                if prev_top_quads is None:
                    should_rebalance = True
                elif current_top_quads != prev_top_quads:
                    should_rebalance = True
                elif len(stop_loss_exits) > 0:
                    should_rebalance = True  # Force rebalance if stops hit
                else:
                    # Check for EMA crossovers (using yesterday's data for consistency)
                    for ticker in yesterday_ema_status:
                        if ticker in prev_ema_status:
                            if yesterday_ema_status[ticker] != prev_ema_status[ticker]:
                                should_rebalance = True
                                break
                
                # Execute rebalancing if triggered
                if should_rebalance or len(confirmed_entries) > 0:
                    rebalance_count += 1
                    
                    # Identify which quads stayed vs changed (to avoid unnecessary rebalancing)
                    quads_that_stayed = set()
                    if prev_top_quads is not None and current_top_quads != prev_top_quads:
                        prev_set = set(prev_top_quads)
                        current_set = set(current_top_quads)
                        quads_that_stayed = prev_set & current_set  # Intersection
                    
                    # Build reverse mapping: ticker -> quads it belongs to
                    ticker_to_quads = {}
                    for quad, allocations in QUAD_ALLOCATIONS.items():
                        for ticker in allocations.keys():
                            if ticker not in ticker_to_quads:
                                ticker_to_quads[ticker] = []
                            ticker_to_quads[ticker].append(quad)
                    
                    # First, apply confirmed entries
                    for ticker, weight in confirmed_entries.items():
                        actual_positions[ticker] = weight
                        # Record entry price, date, and ATR for stop loss tracking
                        # Use OPEN price (no look-ahead bias) - entry executes at open
                        if self.atr_stop_loss is not None and date in self.open_data.index:
                            entry_prices[ticker] = self.open_data.loc[date, ticker]
                            entry_dates[ticker] = date
                            if ticker in self.atr_data.columns:
                                # Use ATR from SIGNAL date (yesterday) for stop calculation
                                prev_date = target_weights.index[target_weights.index.get_loc(date) - 1]
                                entry_atrs[ticker] = self.atr_data.loc[prev_date, ticker]
                    
                    # Now handle the rest of the rebalancing
                    for ticker in target_weights.columns:
                        target_weight = current_targets[ticker]
                        current_position = actual_positions[ticker]
                        position_delta = abs(target_weight - current_position)
                        
                        # Check if this ticker belongs to a quad that stayed in top 2
                        ticker_in_stable_quad = False
                        if ticker in ticker_to_quads and len(quads_that_stayed) > 0:
                            for quad in ticker_to_quads[ticker]:
                                if quad in quads_that_stayed:
                                    ticker_in_stable_quad = True
                                    break
                        
                        if target_weight == 0 and current_position > 0:
                            # Exit immediately (no lag)
                            actual_positions[ticker] = 0
                            # Clear entry tracking
                            if ticker in entry_prices:
                                del entry_prices[ticker]
                            if ticker in entry_dates:
                                del entry_dates[ticker]
                            if ticker in entry_atrs:
                                del entry_atrs[ticker]
                        elif target_weight > 0 and current_position == 0:
                            # New entry - add to pending (wait for confirmation using TOMORROW's EMA)
                            if ticker not in confirmed_entries:  # Don't re-add if just confirmed
                                pending_entries[ticker] = target_weight
                        elif target_weight > 0 and current_position > 0:
                            # Already holding - check if in stable quad
                            if ticker_in_stable_quad:
                                # Ticker is in a quad that stayed in top 2 - DON'T rebalance
                                trades_skipped += 1
                            elif position_delta > MIN_TRADE_THRESHOLD:
                                # Not in stable quad + delta exceeds threshold - rebalance
                                actual_positions[ticker] = target_weight
                            else:
                                # Small delta - skip
                                trades_skipped += 1
                
                # Update tracking variables (use yesterday's for consistency in change detection)
                prev_top_quads = current_top_quads
                prev_ema_status = yesterday_ema_status
            
            # Calculate daily P&L with REALISTIC EXECUTION TIMING
            # =====================================================
            # Overnight (prev close to today open): OLD positions
            # Intraday (today open to today close): NEW positions (if rebalanced)
            # 
            # This accounts for:
            # 1. Gap risk: We hold old positions through the overnight gap
            # 2. Execution lag: New positions start from today's OPEN, not close
            # =====================================================
            
            daily_return = 0
            
            for ticker in actual_positions.index:
                if ticker not in self.price_data.columns:
                    continue
                if ticker not in self.open_data.columns:
                    continue
                    
                old_position = prev_positions[ticker]
                new_position = actual_positions[ticker]
                
                # Get prices
                prev_close = self.price_data.loc[prev_date, ticker]
                today_open = self.open_data.loc[date, ticker]
                today_close = self.price_data.loc[date, ticker]
                
                if pd.isna(prev_close) or pd.isna(today_open) or pd.isna(today_close):
                    continue
                
                # OVERNIGHT RETURN (prev close to today open): Exposed at OLD position
                overnight_return = (today_open / prev_close - 1)
                daily_return += old_position * overnight_return
                
                # INTRADAY RETURN (today open to today close): Exposed at NEW position
                intraday_return = (today_close / today_open - 1)
                daily_return += new_position * intraday_return
            
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + daily_return)
            
            # Calculate trading costs (1 bp per leg)
            # Cost applied on notional value of position changes
            if should_rebalance or len(confirmed_entries) > 0:
                daily_costs = 0.0
                for ticker in actual_positions.index:
                    position_change = abs(actual_positions[ticker] - prev_positions[ticker])
                    if position_change > 0.0001:  # Ignore tiny changes
                        # Notional traded = position change * portfolio value
                        notional_traded = position_change * portfolio_value.iloc[i]
                        # Cost = notional * cost per leg (1 bp = 0.0001)
                        cost = notional_traded * (COST_PER_LEG_BPS / 10000)
                        daily_costs += cost
                
                # Subtract costs from portfolio value
                portfolio_value.iloc[i] -= daily_costs
                total_costs += daily_costs
            
            # Update previous positions for next iteration
            prev_positions = actual_positions.copy()
        
        self.portfolio_value = portfolio_value
        self.total_trading_costs = total_costs
        self.entry_prices = entry_prices  # Current open positions entry prices
        self.entry_dates = entry_dates    # Current open positions entry dates
        self.entry_atrs = entry_atrs      # Current open positions entry ATRs
        
        print(f"  Total rebalances: {rebalance_count} (out of {len(target_weights)-1} days)")
        print(f"  Entries confirmed: {entries_confirmed}")
        print(f"  Entries rejected: {entries_rejected}")
        print(f"  Rejection rate: {entries_rejected / (entries_confirmed + entries_rejected) * 100:.1f}%")
        print(f"  Trades skipped (< 5% delta): {trades_skipped}")
        if self.atr_stop_loss is not None:
            print(f"  Stop losses hit: {stops_hit}")
        print(f"  Trading costs: ${total_costs:,.2f} ({total_costs / self.initial_capital * 100:.2f}% of initial capital)")
        
        # Generate results
        results = self.generate_results()
        
        print("\n" + "=" * 70)
        print("BACKTEST COMPLETE")
        print("=" * 70)
        
        return results
    
    def generate_results(self):
        """Calculate performance metrics"""
        total_return = (self.portfolio_value.iloc[-1] / self.initial_capital - 1) * 100
        
        daily_returns = self.portfolio_value.pct_change().dropna()
        annual_return = ((1 + daily_returns.mean()) ** 252 - 1) * 100
        annual_vol = daily_returns.std() * np.sqrt(252) * 100
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        cummax = self.portfolio_value.expanding().max()
        drawdown = (self.portfolio_value - cummax) / cummax * 100
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': self.portfolio_value.iloc[-1]
        }
    
    def print_annual_breakdown(self):
        """Print annual performance breakdown"""
        returns = self.portfolio_value.pct_change()
        
        print("\n" + "=" * 70)
        print("ANNUAL PERFORMANCE BREAKDOWN")
        print("=" * 70)
        print(f"{'Year':<8}{'Return':<12}{'Sharpe':<12}{'MaxDD':<12}{'Win%':<12}{'Days':<8}")
        print("-" * 70)
        
        for year in returns.index.year.unique():
            year_returns = returns[returns.index.year == year]
            
            if len(year_returns) < 10:
                continue
            
            year_return = (1 + year_returns).prod() - 1
            year_sharpe = year_returns.mean() / year_returns.std() * np.sqrt(252) if year_returns.std() > 0 else 0
            
            year_values = self.portfolio_value[self.portfolio_value.index.year == year]
            year_cummax = year_values.expanding().max()
            year_dd = ((year_values - year_cummax) / year_cummax).min()
            
            win_rate = (year_returns > 0).sum() / len(year_returns)
            
            print(f"{year:<8}{year_return*100:>10.2f}%  {year_sharpe:>10.2f}  "
                  f"{year_dd*100:>10.2f}%  {win_rate*100:>10.1f}%  {len(year_returns):>6}")
        
        print("=" * 70)
    
    def print_spy_comparison(self):
        """Compare strategy to SPY buy-and-hold"""
        # Download SPY data with a buffer
        spy_start = self.portfolio_value.index[0] - timedelta(days=5)
        spy_end = self.portfolio_value.index[-1] + timedelta(days=1)
        
        try:
            spy_data = yf.download('SPY', start=spy_start, end=spy_end, progress=False)
            
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_prices = spy_data['Close'].iloc[:, 0] if isinstance(spy_data['Close'], pd.DataFrame) else spy_data['Close']
            else:
                spy_prices = spy_data['Close']
            
            # Align SPY with portfolio dates
            spy_prices = spy_prices.reindex(self.portfolio_value.index, method='ffill').fillna(method='bfill')
            
            # Calculate SPY returns
            spy_returns = spy_prices.pct_change().dropna()
            spy_total_return = (spy_prices.iloc[-1] / spy_prices.iloc[0] - 1) * 100
            spy_annual_return = ((1 + spy_returns.mean()) ** 252 - 1) * 100
            spy_vol = spy_returns.std() * np.sqrt(252) * 100
            spy_sharpe = spy_annual_return / spy_vol if spy_vol > 0 else 0
            
            spy_cummax = spy_prices.expanding().max()
            spy_dd = ((spy_prices - spy_cummax) / spy_cummax * 100).min()
            
            # Strategy metrics
            strat_returns = self.portfolio_value.pct_change().dropna()
            strat_total = (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0] - 1) * 100
            strat_annual = ((1 + strat_returns.mean()) ** 252 - 1) * 100
            strat_vol = strat_returns.std() * np.sqrt(252) * 100
            strat_sharpe = strat_annual / strat_vol if strat_vol > 0 else 0
            
            strat_cummax = self.portfolio_value.expanding().max()
            strat_dd = ((self.portfolio_value - strat_cummax) / strat_cummax * 100).min()
            
            print("\n" + "=" * 70)
            print("COMPARISON VS S&P 500 (SPY Buy-and-Hold)")
            print("=" * 70)
            print(f"{'Metric':<30}{'Strategy':>15}{'SPY':>15}{'Diff':>15}")
            print("-" * 70)
            print(f"{'Total Return':<30}{strat_total:>14.2f}%{spy_total_return:>14.2f}%{strat_total-spy_total_return:>14.2f}%")
            print(f"{'Annualized Return':<30}{strat_annual:>14.2f}%{spy_annual_return:>14.2f}%{strat_annual-spy_annual_return:>14.2f}%")
            print(f"{'Volatility':<30}{strat_vol:>14.2f}%{spy_vol:>14.2f}%{strat_vol-spy_vol:>14.2f}%")
            print(f"{'Sharpe Ratio':<30}{strat_sharpe:>15.2f}{spy_sharpe:>15.2f}{strat_sharpe-spy_sharpe:>15.2f}")
            print(f"{'Max Drawdown':<30}{strat_dd:>14.2f}%{spy_dd:>14.2f}%{strat_dd-spy_dd:>14.2f}%")
            print()
            print(f"{'Alpha (vs SPY)':<30}{strat_annual-spy_annual_return:>14.2f}%")
            print(f"{'Outperformance':<30}{strat_total-spy_total_return:>14.2f}%")
            print("=" * 70)
            
        except Exception as e:
            print(f"\nCould not compare to SPY: {e}")
    
    def plot_results(self):
        """Plot portfolio performance"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Portfolio value
        ax1.plot(self.portfolio_value.index, self.portfolio_value.values, 
                linewidth=2, color='purple', label='Portfolio Value')
        ax1.set_title('Macro Quadrant Rotation Strategy - Production Version', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Drawdown
        cummax = self.portfolio_value.expanding().max()
        drawdown = (self.portfolio_value - cummax) / cummax * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        alpha=0.3, color='red', label='Drawdown')
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "=" * 70)
        print("📊 Chart displayed")
        print("=" * 70)


if __name__ == "__main__":
    # Configuration
    INITIAL_CAPITAL = 50000
    LOOKBACK_DAYS = 50
    EMA_PERIOD = 50
    VOL_LOOKBACK = 30
    BACKTEST_YEARS = 5
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * BACKTEST_YEARS + 200)
    
    print("\n" + "=" * 70)
    print("MACRO QUADRANT ROTATION STRATEGY - PRODUCTION VERSION")
    print("=" * 70)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Momentum Lookback: {LOOKBACK_DAYS} days")
    print(f"EMA Trend Filter: {EMA_PERIOD}-day")
    print(f"Volatility Lookback: {VOL_LOOKBACK} days")
    print(f"Backtest Period: ~{BACKTEST_YEARS} years")
    print(f"Leverage: UNIFORM (All Quads=150%)")
    print(f"Entry Confirmation: 1-day lag using live EMA")
    print("=" * 70)
    print()
    
    # Run backtest
    backtest = QuadrantPortfolioBacktest(start_date, end_date, INITIAL_CAPITAL, 
                                         LOOKBACK_DAYS, EMA_PERIOD, VOL_LOOKBACK)
    results = backtest.run_backtest()
    
    # Print results
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Initial Capital...................................  ${INITIAL_CAPITAL:>12,}")
    print(f"Final Capital.....................................  ${results['final_value']:>12,.2f}")
    print(f"Total Return......................................  {results['total_return']:>12.2f}%")
    print(f"Annualized Return.................................  {results['annual_return']:>12.2f}%")
    print(f"Annualized Volatility.............................  {results['annual_vol']:>12.2f}%")
    print(f"Sharpe Ratio......................................  {results['sharpe']:>12.2f}")
    print(f"Maximum Drawdown..................................  {results['max_drawdown']:>12.2f}%")
    print(f"Start Date........................................  {backtest.portfolio_value.index[0].strftime('%Y-%m-%d'):>15}")
    print(f"End Date..........................................  {backtest.portfolio_value.index[-1].strftime('%Y-%m-%d'):>15}")
    print(f"Trading Days......................................  {len(backtest.portfolio_value):>15,}")
    print(f"Total Trading Costs...............................  ${backtest.total_trading_costs:>12,.2f}")
    print(f"Costs as % of Initial Capital....................  {backtest.total_trading_costs / INITIAL_CAPITAL * 100:>14.2f}%")
    print(f"Costs as % of Final Capital.......................  {backtest.total_trading_costs / results['final_value'] * 100:>14.2f}%")
    print("=" * 70)
    
    # Annual breakdown
    backtest.print_annual_breakdown()
    
    # SPY comparison
    backtest.print_spy_comparison()
    
    backtest.plot_results()
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE - PRODUCTION VERSION")
    print("=" * 70)
    print("\nStrategy: Macro Quadrant Rotation with Entry Confirmation")
    print("Key Features:")
    print("  - Quad signals: T-1 lag (prevent forward-looking bias)")
    print("  - Entry confirmation: T+0 (live EMA filter)")
    print("  - Volatility chasing: 30-day lookback")
    print("  - Uniform leverage: All quads=1.5x")
    print("=" * 70)

