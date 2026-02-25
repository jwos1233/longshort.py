"""
Signal Generator for Macro Quadrant Strategy
==============================================

Generates trading signals based on macro regime detection.

Strategy:
- Identifies top 2 quadrants using 50-day momentum
- Weights assets within quadrants by 30-day volatility (volatility chasing)
- Filters: 50-day EMA (only allocate above EMA)
- Asymmetric leverage: Q1=1.5x, Q2/Q3/Q4=1.0x
- Entry confirmation: 1-day lag on live EMA status
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from typing import Dict, Tuple
import sys
from config import (
    QUAD_ALLOCATIONS,
    BTC_PROXY_BASKET,
    BTC_PROXY_MAX_POSITIONS,
    EXPAND_TO_CONSTITUENTS,
    TOP_CONSTITUENTS_PER_ETF,
)

# Windows consoles sometimes default to a legacy codepage (cp1252) that can't print
# common Unicode symbols used in logs. Ensure UTF-8 output to avoid crashes.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Quadrant indicators for momentum scoring
QUAD_INDICATORS = {
    'Q1': ['QQQ', 'VUG', 'IWM', 'BTC-USD'],
    'Q2': ['XLE', 'DBC'],
    'Q3': ['GLD', 'LIT'],
    'Q4': ['TLT', 'XLU', 'VIXY']
}


class SignalGenerator:
    """Generate live trading signals for macro quadrant rotation strategy"""
    
    def __init__(self, momentum_days=20, ema_period=50, vol_lookback=30, max_positions=10,
                 atr_stop_loss=2.0, atr_period=14, ema_smoothing_period=20):
        self.momentum_days = momentum_days
        self.ema_period = ema_period
        self.vol_lookback = vol_lookback
        self.max_positions = max_positions  # Top 10 positions (optimal from backtesting)
        self.atr_stop_loss = atr_stop_loss  # ATR 2.0x stop loss (optimal from backtesting)
        self.atr_period = atr_period  # 14-day ATR
        self.ema_smoothing_period = ema_smoothing_period  # EMA smoothing for quad scores
        
        # Leverage by quadrant
        self.quad_leverage = {
            'Q1': 1.5,  # Goldilocks - overweight
            'Q2': 1.0,  # Reflation
            'Q3': 1.0,  # Stagflation
            'Q4': 1.0   # Deflation
        }
    
    def fetch_market_data(self, lookback_days=150):
        """
        Fetch market data for all tickers
        
        Args:
            lookback_days: Number of days to fetch (default 150 for buffers)
        
        Returns:
            DataFrame with price data
        """
        # Get all unique tickers
        all_tickers = set()
        for quad_assets in QUAD_ALLOCATIONS.values():
            all_tickers.update(quad_assets.keys())
        for indicators in QUAD_INDICATORS.values():
            all_tickers.update(indicators)

        # Ensure BTC proxy basket tickers are fetched so we can replace BTC-USD at execution time
        all_tickers.update(BTC_PROXY_BASKET.keys())

        # If expanding to constituents, add all constituent tickers for equity ETFs so we have price/vol data
        if EXPAND_TO_CONSTITUENTS:
            from etf_mapper import get_constituent_tickers_for_universe
            constituent_tickers = get_constituent_tickers_for_universe(all_tickers)
            all_tickers.update(constituent_tickers)

        all_tickers = sorted(list(all_tickers))
        
        # Use yesterday's date to ensure consistent data availability
        # This prevents issues where same-day data isn't finalized yet when run right after close
        # When run on T+1, it will use T's finalized data (which is what we want)
        today = date.today()
        end_date = datetime.combine(today - timedelta(days=1), datetime.min.time())
        start_date = end_date - timedelta(days=lookback_days)
        
        print(f"Fetching data for {len(all_tickers)} tickers...")
        print(f"  Date range: {start_date.date()} to {end_date.date()} (using finalized close prices)")
        
        price_series = []
        last_available_dates = []
        
        for ticker in all_tickers:
            try:
                # Use period='6mo' to get recent data, which is more reliable than start/end dates
                # This ensures we get the most recent available finalized data
                data = yf.download(ticker, period='6mo',
                                 progress=False, auto_adjust=True)
                if len(data) > 0:
                    # Handle MultiIndex columns from newer yfinance versions
                    if isinstance(data.columns, pd.MultiIndex):
                        if 'Close' not in data.columns.get_level_values(0):
                            continue
                        series = data['Close']
                    elif 'Close' in data.columns:
                        series = data['Close']
                    else:
                        continue
                    # Squeeze DataFrame to Series if needed
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[:, 0]
                    series = series.copy()
                    # Filter to our desired date range (but keep all recent data)
                    series = series[series.index.date >= start_date.date()]
                    # Only include data up to our target end_date (yesterday)
                    # This ensures consistency - always uses finalized close prices
                    series = series[series.index.date <= end_date.date()]
                    if len(series) > 0:
                        series.name = ticker
                        price_series.append(series)
                        last_available_dates.append(series.index[-1])
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
        
        if not price_series:
            raise ValueError("No price data loaded!")
        
        df = pd.concat(price_series, axis=1)
        df = df.ffill().bfill()
        
        # Determine actual last available date (most common date across all tickers)
        if last_available_dates:
            # Convert to dates if they're Timestamps
            dates_only = [d.date() if hasattr(d, 'date') else d for d in last_available_dates]
            actual_last_date = max(set(dates_only))
            print(f"✓ Loaded {len(df.columns)} tickers, {len(df)} days")
            print(f"  Last available price date: {actual_last_date}")
        else:
            print(f"✓ Loaded {len(df.columns)} tickers, {len(df)} days")
        
        return df
    
    def calculate_quadrant_scores(self, price_data: pd.DataFrame, apply_smoothing: bool = True) -> pd.Series:
        """
        Calculate momentum scores for each quadrant
        
        Args:
            price_data: DataFrame with price data
            apply_smoothing: If True, apply EMA smoothing to scores
        
        Returns:
            Series with quad scores for today (smoothed if apply_smoothing=True)
        """
        # Calculate raw scores for all dates (needed for EMA smoothing)
        raw_scores_df = pd.DataFrame(index=price_data.index, columns=list(QUAD_INDICATORS.keys()))
        
        for quad, indicators in QUAD_INDICATORS.items():
            quad_score_series = pd.Series(index=price_data.index, dtype=float)
            
            for date in price_data.index:
                quad_scores_list = []
                for ticker in indicators:
                    if ticker in price_data.columns:
                        momentum = price_data[ticker].pct_change(self.momentum_days).loc[date] * 100
                        if pd.notna(momentum):
                            quad_scores_list.append(momentum)
                
                if quad_scores_list:
                    quad_score_series.loc[date] = np.mean(quad_scores_list)
                else:
                    quad_score_series.loc[date] = 0.0
            
            raw_scores_df[quad] = quad_score_series
        
        # Apply EMA smoothing if requested
        if apply_smoothing and self.ema_smoothing_period and self.ema_smoothing_period > 0:
            smoothed_scores_df = pd.DataFrame(index=price_data.index, columns=list(QUAD_INDICATORS.keys()))
            for quad in QUAD_INDICATORS.keys():
                smoothed_scores_df[quad] = raw_scores_df[quad].ewm(
                    span=self.ema_smoothing_period, 
                    adjust=False
                ).mean()
            
            # Return smoothed scores for last date
            final_scores = {}
            for quad in QUAD_INDICATORS.keys():
                final_scores[quad] = smoothed_scores_df[quad].iloc[-1]
        else:
            # Return raw scores for last date
            final_scores = {}
            for quad in QUAD_INDICATORS.keys():
                final_scores[quad] = raw_scores_df[quad].iloc[-1]
        
        return pd.Series(final_scores).sort_values(ascending=False)
    
    def get_top_quadrants(self, quad_scores: pd.Series) -> Tuple[str, str]:
        """Get top 2 quadrants"""
        top_quads = quad_scores.index[:2].tolist()
        return top_quads[0], top_quads[1]
    
    def calculate_target_weights(self, price_data: pd.DataFrame, 
                                 top1: str, top2: str) -> Tuple[Dict[str, float], Dict[str, dict]]:
        """
        Calculate target portfolio weights
        
        Returns:
            Tuple of:
            - Dictionary of {ticker: weight} where weights sum to ~2.5 (if Q1 active)
            - Dictionary of {ticker: {'price': float, 'ema': float, 'quadrant': str}} for excluded assets
        """
        # Calculate EMA
        ema_data = price_data.ewm(span=self.ema_period, adjust=False).mean()
        
        # Calculate volatility
        returns = price_data.pct_change()
        volatility_data = returns.rolling(window=self.vol_lookback).std() * np.sqrt(252)
        
        final_weights = {}
        excluded_below_ema = {}  # Track assets excluded due to being below EMA
        
        for quad in [top1, top2]:
            # Get leverage for this quad
            quad_leverage = self.quad_leverage[quad]
            
            # Get tickers in this quad
            quad_tickers = [t for t in QUAD_ALLOCATIONS[quad].keys() 
                          if t in price_data.columns]
            
            if not quad_tickers:
                continue
            
            # Get current volatilities
            quad_vols = {}
            for ticker in quad_tickers:
                vol = volatility_data[ticker].iloc[-1]
                if pd.notna(vol) and vol > 0:
                    quad_vols[ticker] = vol
            
            if not quad_vols:
                continue
            
            # Volatility chasing weights
            total_vol = sum(quad_vols.values())
            vol_weights = {t: (v / total_vol) * quad_leverage 
                          for t, v in quad_vols.items()}
            
            # Apply EMA filter
            for ticker, weight in vol_weights.items():
                current_price = price_data[ticker].iloc[-1]
                current_ema = ema_data[ticker].iloc[-1]
                
                if pd.notna(current_price) and pd.notna(current_ema):
                    if current_price > current_ema:
                        # Pass EMA filter
                        if ticker in final_weights:
                            final_weights[ticker] += weight
                        else:
                            final_weights[ticker] = weight
                    else:
                        # Below EMA - exclude from weights but track it
                        excluded_below_ema[ticker] = {
                            'price': float(current_price),
                            'ema': float(current_ema),
                            'quadrant': quad,
                            'would_be_weight': weight
                        }

        # Replace BTC-USD weight with a proxy basket (if present).
        # Key design: select up to BTC_PROXY_MAX_POSITIONS proxies by the same "vol chasing"
        # idea (higher vol = higher weight), using BTC_PROXY_BASKET weights as a prior.
        if 'BTC-USD' in final_weights and BTC_PROXY_BASKET:
            btc_weight = final_weights.pop('BTC-USD')

            # Compute vols for proxy tickers (pure volatility weighting, same as rest of portfolio)
            proxy_vols = {}
            for proxy_ticker in BTC_PROXY_BASKET.keys():
                if proxy_ticker not in price_data.columns:
                    continue
                vol = volatility_data[proxy_ticker].iloc[-1] if proxy_ticker in volatility_data.columns else np.nan
                if pd.isna(vol) or vol <= 0:
                    continue

                # Apply EMA filter at the proxy level
                current_price = price_data[proxy_ticker].iloc[-1]
                current_ema = ema_data[proxy_ticker].iloc[-1] if proxy_ticker in ema_data.columns else np.nan
                if pd.isna(current_price) or pd.isna(current_ema) or current_price <= current_ema:
                    continue

                # Pure volatility weighting (volatility chasing within crypto bucket)
                proxy_vols[proxy_ticker] = float(vol)

            if not proxy_vols:
                # No eligible proxies => BTC sleeve becomes cash
                excluded_below_ema['BTC-USD'] = {
                    'price': float(price_data['BTC-USD'].iloc[-1]) if 'BTC-USD' in price_data.columns else None,
                    'ema': float(ema_data['BTC-USD'].iloc[-1]) if 'BTC-USD' in ema_data.columns else None,
                    'quadrant': 'CRYPTO_PROXY',
                    'would_be_weight': btc_weight,
                    'reason': 'No eligible BTC proxies (data/vol/EMA)',
                }
            else:
                # Keep only top N proxies (by volatility) to avoid fragmenting the portfolio
                n = int(BTC_PROXY_MAX_POSITIONS) if BTC_PROXY_MAX_POSITIONS else 10
                # Sort by volatility (highest first) and take top N
                top_proxies = sorted(proxy_vols.items(), key=lambda x: x[1], reverse=True)[:n]
                total_vol = sum(v for _, v in top_proxies)
                
                if total_vol <= 0:
                    excluded_below_ema['BTC-USD'] = {
                        'price': float(price_data['BTC-USD'].iloc[-1]) if 'BTC-USD' in price_data.columns else None,
                        'ema': float(ema_data['BTC-USD'].iloc[-1]) if 'BTC-USD' in ema_data.columns else None,
                        'quadrant': 'CRYPTO_PROXY',
                        'would_be_weight': btc_weight,
                        'reason': 'BTC proxy volatilities invalid',
                    }
                else:
                    # Weight by volatility (volatility chasing within crypto bucket)
                    for proxy_ticker, vol in top_proxies:
                        proxy_weight = btc_weight * (vol / total_vol)
                        final_weights[proxy_ticker] = final_weights.get(proxy_ticker, 0.0) + proxy_weight
        
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
        
        # ENFORCE: Never return more than max_positions
        if self.max_positions and len(final_weights) > self.max_positions:
            sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
            final_weights = dict(sorted_weights[:self.max_positions])
            print(f"⚠️ WARNING: Had to force-filter to {self.max_positions} positions!")
        
        return final_weights, excluded_below_ema
    
    def generate_signals(self) -> Dict:
        """
        Generate current trading signals
        
        Returns:
            Dictionary with:
            - top_quadrants: (Q1, Q2) tuple
            - quadrant_scores: Series of all quad scores
            - target_weights: Dict of {ticker: weight}
            - current_regime: str description
            - timestamp: datetime
        """
        print("\n" + "="*60)
        print("GENERATING SIGNALS")
        print("="*60)
        
        # Fetch data
        price_data = self.fetch_market_data(lookback_days=150)
        
        # Calculate and store EMA data
        self.price_data = price_data
        self.ema_data = price_data.ewm(span=self.ema_period, adjust=False).mean()
        
        # Calculate quadrant scores (with EMA smoothing if enabled)
        apply_smoothing = self.ema_smoothing_period and self.ema_smoothing_period > 0
        if apply_smoothing:
            print(f"\nCalculating quadrant scores with {self.ema_smoothing_period}-period EMA smoothing...")
        quad_scores = self.calculate_quadrant_scores(price_data, apply_smoothing=apply_smoothing)
        if apply_smoothing:
            print(f"✓ Smoothed scores calculated")
        top1, top2 = self.get_top_quadrants(quad_scores)
        
        print(f"\nQuadrant Scores:")
        for quad in quad_scores.index:
            print(f"  {quad}: {quad_scores[quad]:>7.2f}%")
        
        print(f"\n🎯 Top 2 Quadrants: {top1}, {top2}")
        
        # Calculate target weights (ETF-level)
        target_weights, excluded_below_ema = self.calculate_target_weights(price_data, top1, top2)

        # Optionally expand equity ETF weights to constituent stocks (vol chasing + top N per ETF)
        if EXPAND_TO_CONSTITUENTS and target_weights:
            from constituent_expander import expand_etf_weights_to_constituents
            target_weights = expand_etf_weights_to_constituents(
                target_weights,
                price_data,
                vol_lookback=self.vol_lookback,
                ema_period=self.ema_period,
                top_n_per_etf=TOP_CONSTITUENTS_PER_ETF,
                as_of_date=None,
                verbose=True,
            )

        # Calculate ATR for stop losses
        atr_data = {}
        if self.atr_stop_loss is not None and len(target_weights) > 0:
            print(f"\n📐 Calculating ATR for stop losses ({self.atr_period}-day, {self.atr_stop_loss}x)...")
            daily_returns = price_data.pct_change().abs()
            atr = daily_returns.rolling(window=self.atr_period).mean() * price_data
            
            for ticker in target_weights.keys():
                if ticker in atr.columns:
                    atr_value = atr[ticker].iloc[-1]
                    if pd.notna(atr_value):
                        atr_data[ticker] = float(atr_value)
        
        # Calculate total leverage
        total_leverage = sum(target_weights.values())
        
        print(f"\n📊 Target Portfolio (Top {self.max_positions} Positions):")
        print(f"  Total leverage: {total_leverage:.2f}x")
        print(f"  Number of positions: {len(target_weights)}")
        
        if target_weights:
            print(f"\n  ALL POSITIONS (sorted by weight):")
            print(f"  {'Ticker':<8} {'Weight':<10} {'Notional ($10k)':<15} {'Quadrant':<10}")
            print(f"  {'-'*8} {'-'*10} {'-'*15} {'-'*10}")
            
            sorted_weights = sorted(target_weights.items(), key=lambda x: x[1], reverse=True)
            for ticker, weight in sorted_weights:
                # Determine which quadrant(s) this ticker belongs to
                quads = []
                for q, assets in QUAD_ALLOCATIONS.items():
                    if ticker in assets:
                        quads.append(q)
                
                quad_str = '+'.join(quads) if quads else ''
                
                # Calculate notional value for $10k account
                notional_10k = weight * 10000
                
                print(f"  {ticker:<8} {weight*100:>8.2f}%  ${notional_10k:>12,.2f}  {quad_str:<10}")
        
        # Print excluded assets if any
        if excluded_below_ema:
            print(f"\n  ⚠️ EXCLUDED (Below EMA):")
            print(f"  {'Ticker':<8} {'Price':<12} {'EMA':<12} {'% Below':<10} {'Quadrant':<10}")
            print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
            
            sorted_excluded = sorted(excluded_below_ema.items(), 
                                    key=lambda x: x[1].get('would_be_weight', 0), 
                                    reverse=True)
            for ticker, info in sorted_excluded:
                price = info.get('price', 0)
                ema_val = info.get('ema', 0)
                pct_below = ((price - ema_val) / ema_val * 100) if ema_val > 0 else 0
                quad = info.get('quadrant', '')
                print(f"  {ticker:<8} ${price:>10.2f}  ${ema_val:>10.2f}  {pct_below:>8.2f}%  {quad:<10}")
        
        # Get the date of the last price data point (the date prices are from)
        price_date = price_data.index[-1] if len(price_data) > 0 else datetime.now().date()
        if isinstance(price_date, pd.Timestamp):
            price_date = price_date.date()
        
        # Get UTC timestamp for when analysis was run
        analysis_timestamp_utc = datetime.utcnow()
        
        return {
            'top_quadrants': (top1, top2),
            'quadrant_scores': quad_scores,
            'target_weights': target_weights,
            'excluded_below_ema': excluded_below_ema,
            'current_regime': f"{top1} + {top2}",
            'timestamp': datetime.now(),
            'price_date': price_date,  # Date of the price data used
            'analysis_timestamp_utc': analysis_timestamp_utc,  # UTC time when analysis was run
            'total_leverage': total_leverage,
            'atr_data': atr_data
        }


if __name__ == "__main__":
    # Test signal generation
    sg = SignalGenerator()
    signals = sg.generate_signals()
    
    print("\n" + "="*60)
    print("SIGNAL GENERATION COMPLETE")
    print("="*60)
    print(f"Timestamp: {signals['timestamp']}")
    print(f"Regime: {signals['current_regime']}")
    print(f"Total Leverage: {signals['total_leverage']:.2f}x")
    print(f"Positions: {len(signals['target_weights'])}")
    
    # Export to CSV-friendly format (sorted by weight, largest to smallest)
    print("\n📋 CSV Export Format (Sorted by Weight):")
    print("Ticker,Weight(%),Quadrant(s)")
    sorted_by_weight = sorted(signals['target_weights'].items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_by_weight:
        quads = []
        for q, assets in QUAD_ALLOCATIONS.items():
            if ticker in assets:
                quads.append(q)
        quad_str = '+'.join(quads) if quads else ''
        print(f"{ticker},{weight*100:.2f}%,{quad_str}")

