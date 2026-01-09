"""
Simplified Live Trader - Single Run
====================================

Simplified version that does everything in one execution:
1. Generate signals from yesterday's close (T-1)
2. Confirm with today's EMA (T)
3. Execute trades immediately

Matches backtest logic exactly without the complexity of two-step process.

Usage:
    # Dry run (see what it would do)
    
    python live_trader_simple.py --port 4001
    
    # Live execution
    python live_trader_simple.py --port 4001 --live
    
    # Disable Telegram
    python live_trader_simple.py --port 4001 --live --no-telegram
"""

from signal_generator import SignalGenerator
from ib_executor import IBExecutor
from position_manager import PositionManager
from telegram_notifier import get_notifier
from datetime import datetime
import yfinance as yf
import pandas as pd


class SimpleLiveTrader:
    """Simplified live trader - generate, confirm, and execute in one run"""
    
    def __init__(self, ib_port=4001, dry_run=True, enable_telegram=True):
        """
        Initialize trader
        
        Args:
            ib_port: IB Gateway port (4001=live, 4002=paper)
            dry_run: If True, show what would happen but don't trade
            enable_telegram: Send notifications
        """
        self.signal_gen = SignalGenerator(
            momentum_days=20, 
            ema_period=50, 
            vol_lookback=30, 
            max_positions=10,
            atr_stop_loss=2.0,
            atr_period=14,
            ema_smoothing_period=20  # EMA smoothing for quad scores
        )
        self.ib_port = ib_port
        self.dry_run = dry_run
        self.telegram = get_notifier() if enable_telegram else None
    
    def get_current_ema_status(self, tickers: list, ema_period: int = 50) -> dict:
        """
        Get CURRENT EMA status for confirmation
        
        This uses TODAY's/current data for EMA confirmation,
        matching the backtest's T+0 confirmation logic.
        
        Args:
            tickers: List of tickers to check
            ema_period: EMA period (default 50)
        
        Returns:
            Dict of {ticker: (current_price, current_ema, is_above_ema)}
        """
        print(f"\nFetching current market data for {len(tickers)} tickers...")
        
        # Fetch recent data (need enough for EMA calculation)
        price_data = yf.download(
            tickers, 
            period='3mo',  # 3 months to ensure enough data for 50-day EMA
            progress=False,
            auto_adjust=True
        )['Close']
        
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame(name=tickers[0])
        
        print(f"+ Loaded data for {len(price_data.columns)} tickers")
        
        # Calculate current EMA for each ticker
        ema_status = {}
        
        for ticker in tickers:
            if ticker not in price_data.columns:
                print(f"  WARNING: No data for {ticker}")
                continue
            
            prices = price_data[ticker].dropna()
            
            if len(prices) < ema_period:
                print(f"  WARNING: Not enough data for {ticker}")
                continue
            
            # Calculate 50-day EMA
            ema = prices.ewm(span=ema_period, adjust=False).mean()
            
            # Get current (most recent) values
            current_price = prices.iloc[-1]
            current_ema = ema.iloc[-1]
            is_above = current_price > current_ema
            
            ema_status[ticker] = {
                'current_price': current_price,
                'current_ema': current_ema,
                'is_above_ema': is_above
            }
        
        return ema_status
    
    def confirm_entries(self, target_weights: dict, ema_status: dict) -> tuple:
        """
        Confirm entries using current EMA
        
        Args:
            target_weights: Dict of {ticker: weight} from signals
            ema_status: Current EMA status from get_current_ema_status()
        
        Returns:
            (confirmed_weights, rejected_tickers) tuple
        """
        print("\n" + "="*70)
        print(f"ENTRY CONFIRMATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print(f"Pending entries: {len(target_weights)}\n")
        
        confirmed = {}
        rejected = {}
        
        for ticker, weight in target_weights.items():
            if ticker not in ema_status:
                print(f"  WARNING: {ticker}: No EMA data - SKIP")
                rejected[ticker] = weight
                continue
            
            status = ema_status[ticker]
            current_price = status['current_price']
            current_ema = status['current_ema']
            is_above = status['is_above_ema']
            
            if is_above:
                print(f"  ✅ {ticker}: ${current_price:.2f} > ${current_ema:.2f} EMA - CONFIRMED")
                confirmed[ticker] = weight
            else:
                print(f"  ❌ {ticker}: ${current_price:.2f} < ${current_ema:.2f} EMA - REJECTED")
                rejected[ticker] = weight
        
        # Display summary
        total = len(confirmed) + len(rejected)
        rejection_rate = (len(rejected) / total * 100) if total > 0 else 0
        
        print("\n" + "="*70)
        print("CONFIRMATION SUMMARY")
        print("="*70)
        print(f"Confirmed: {len(confirmed)}")
        print(f"Rejected:  {len(rejected)}")
        print(f"Rejection rate: {rejection_rate:.1f}%")
        print("="*70)
        
        return confirmed, rejected
    
    def run(self):
        """Execute complete trading workflow"""
        print("\n" + "="*70)
        print(f"SIMPLIFIED LIVE TRADER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE TRADING'}")
        print(f"Port: {self.ib_port}")
        print("="*70)
        
        try:
            # STEP 1: Generate signals from YESTERDAY's close (T-1)
            print("\n" + "="*70)
            print("STEP 1: SIGNAL GENERATION (from yesterday's close)")
            print("="*70)
            
            signals = self.signal_gen.generate_signals()
            
            target_weights = signals['target_weights']
            regime = signals['current_regime']
            top_quads = signals['top_quadrants']
            leverage = signals['total_leverage']
            
            print(f"\nSignals generated:")
            print(f"  Regime: {regime}")
            print(f"  Top quadrants: {top_quads[0]}, {top_quads[1]}")
            print(f"  Positions: {len(target_weights)}")
            print(f"  Leverage: {leverage:.2f}x")
            
            if not target_weights:
                print("\nNo positions to enter - exiting")
                return
            
            # STEP 2: Confirm entries using CURRENT EMA (T)
            print("\n" + "="*70)
            print("STEP 2: EMA CONFIRMATION (using current/today's EMA)")
            print("="*70)
            
            tickers = list(target_weights.keys())
            ema_status = self.get_current_ema_status(tickers)
            
            confirmed_weights, rejected = self.confirm_entries(target_weights, ema_status)
            
            if not confirmed_weights:
                print("\nNo entries confirmed - all rejected!")
                if self.telegram:
                    self.telegram.send_error_alert("No entries confirmed - all rejected", "Single Run Trader")
                return
            
            # DRY RUN: Show what would happen
            if self.dry_run:
                print("\n" + "="*70)
                print("DRY RUN MODE - NO TRADES EXECUTED")
                print("="*70)
                print("\nConfirmed positions that WOULD be traded:")
                for ticker, weight in sorted(confirmed_weights.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {ticker}: {weight*100:.2f}%")
                print("\nRejected positions:")
                for ticker in rejected.keys():
                    print(f"  {ticker}")
                print("\nRe-run with --live flag to execute trades")
                return
            
            # STEP 3: Execute trades via IB
            print("\n" + "="*70)
            print("STEP 3: EXECUTION VIA INTERACTIVE BROKERS")
            print("="*70)
            
            # Get ATR data for stops (from signals dict, not signal_gen attribute)
            atr_data = signals.get('atr_data', {})
            
            # Filter to only confirmed tickers
            atr_data = {ticker: atr for ticker, atr in atr_data.items() 
                       if ticker in confirmed_weights}
            
            print(f"\n📐 ATR Data for stops: {len(atr_data)} tickers")
            if atr_data:
                for ticker, atr in list(atr_data.items())[:3]:  # Show first 3
                    print(f"  {ticker}: ${atr:.2f}")
                if len(atr_data) > 3:
                    print(f"  ... and {len(atr_data) - 3} more")
            else:
                print("  ⚠️ WARNING: No ATR data available - stops will NOT be placed!")
            
            with IBExecutor(port=self.ib_port) as ib_exec:
                if not ib_exec.connected:
                    print("\nERROR: Failed to connect to IB Gateway")
                    print("Make sure IB Gateway is running on port", self.ib_port)
                    if self.telegram:
                        self.telegram.send_error_alert("Failed to connect to IB Gateway", "Single Run Trader")
                    return
                
                print("+ Connected to IB Gateway")
                
                # Get account value
                account_value = ib_exec.get_account_value()
                print(f"+ Account value: ${account_value:,.2f}")
                
                if account_value <= 0:
                    print("\nERROR: Invalid account value")
                    return
                
                # Initialize position manager
                position_manager = PositionManager(ib_exec.ib)
                
                # Get positions before
                positions_before = ib_exec.get_current_positions()
                print(f"+ Current positions: {len(positions_before)}")
                
                # Execute rebalance
                trades = ib_exec.execute_rebalance(
                    confirmed_weights,
                    position_manager=position_manager,
                    atr_data=atr_data
                )
                
                # Get positions after
                positions_after = ib_exec.get_current_positions()
                
                # ENFORCE TOP 10: Close any positions not in confirmed_weights
                positions_to_close = [ticker for ticker in positions_after.keys() 
                                     if ticker not in confirmed_weights]
                if positions_to_close:
                    print(f"\n⚠️ Found {len(positions_to_close)} positions not in top 10 - closing...")
                    for ticker in positions_to_close:
                        try:
                            contract = ib_exec.create_cfd_contract(ticker)
                            if contract:
                                quantity = int(abs(positions_after[ticker]))
                                if quantity > 0:
                                    action = 'SELL' if positions_after[ticker] > 0 else 'BUY'
                                    print(f"  Closing {ticker} ({quantity} shares)...")
                                    trade = ib_exec.place_order(contract, quantity, action)
                                    if trade:
                                        print(f"    ✓ Closed {ticker}")
                        except Exception as e:
                            print(f"    ✗ Error closing {ticker}: {e}")
                    
                    # Re-fetch positions after cleanup
                    positions_after = ib_exec.get_current_positions()
                
                # Display summary
                print("\n" + "="*70)
                print("EXECUTION COMPLETE")
                print("="*70)
                print(f"Confirmed entries: {len(confirmed_weights)}")
                print(f"Rejected entries: {len(rejected)}")
                print(f"Trades executed: {len(trades) if trades else 0}")
                print(f"Final positions: {len(positions_after)}")
                if len(positions_after) > len(confirmed_weights):
                    print(f"⚠️ WARNING: {len(positions_after)} positions but only {len(confirmed_weights)} confirmed!")
                print("="*70)
                
                # Send Telegram notification
                if self.telegram:
                    try:
                        positions_summary = self._build_positions_summary(
                            positions_before, positions_after
                        )
                        
                        self.telegram.send_morning_alert(
                            confirmed_weights,
                            rejected,
                            trades if trades else [],
                            positions_summary
                        )
                        print("+ Telegram notification sent")
                    except Exception as e:
                        print(f"! Telegram notification failed: {e}")
                
                # Save execution log
                self._save_execution_log(
                    confirmed_weights, rejected, trades,
                    positions_before, positions_after, account_value
                )
                
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            
            if self.telegram:
                try:
                    self.telegram.send_error_alert(str(e), "Single Run Trader")
                except:
                    pass
    
    def _build_positions_summary(self, positions_before: dict, positions_after: dict) -> dict:
        """Build positions summary for Telegram"""
        all_tickers = set(list(positions_before.keys()) + list(positions_after.keys()))
        
        added = []
        removed = []
        adjusted = []
        
        for ticker in sorted(all_tickers):
            before = positions_before.get(ticker, 0)
            after = positions_after.get(ticker, 0)
            
            if before == 0 and after > 0:
                added.append(f"{ticker} (+{after:.0f})")
            elif before > 0 and after == 0:
                removed.append(f"{ticker} (-{before:.0f})")
            elif before != after:
                change = after - before
                adjusted.append(f"{ticker} ({change:+.0f})")
        
        return {
            'added': added,
            'removed': removed,
            'adjusted': adjusted
        }
    
    def _save_execution_log(self, confirmed, rejected, trades, 
                           positions_before, positions_after, account_value):
        """Save execution log to file"""
        filename = f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"EXECUTION LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Account Value: ${account_value:,.2f}\n")
            f.write(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE TRADING'}\n\n")
            
            f.write("CONFIRMATION RESULTS:\n")
            f.write("-"*70 + "\n")
            f.write(f"Confirmed: {len(confirmed)}\n")
            f.write(f"Rejected: {len(rejected)}\n")
            
            if confirmed:
                f.write("\nConfirmed entries:\n")
                for ticker, weight in sorted(confirmed.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {ticker}: {weight*100:.2f}%\n")
            
            if rejected:
                f.write("\nRejected entries:\n")
                for ticker in rejected.keys():
                    f.write(f"  {ticker}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("TRADES EXECUTED:\n")
            f.write("="*70 + "\n")
            
            if trades:
                for i, trade in enumerate(trades, 1):
                    f.write(f"{i}. {trade}\n")
            else:
                f.write("No trades executed\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("POSITIONS:\n")
            f.write("="*70 + "\n")
            f.write("\nBefore:\n")
            for ticker, qty in sorted(positions_before.items()):
                f.write(f"  {ticker}: {qty:.0f}\n")
            
            f.write("\nAfter:\n")
            for ticker, qty in sorted(positions_after.items()):
                f.write(f"  {ticker}: {qty:.0f}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"+ Execution log saved to: {filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Simplified Live Trader - Single Run Execution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (see what would happen)
  python live_trader_simple.py --port 4001
  
  # Live execution
  python live_trader_simple.py --port 4001 --live
  
  # Scheduled execution (via cron)
  35 14 * * 1-5 cd ~/strategy && source venv/bin/activate && python live_trader_simple.py --port 4001 --live

How it works:
  1. Generates signals from yesterday's close (T-1 for quad rankings)
  2. Confirms with today's EMA (T for entry confirmation)
  3. Executes trades immediately
  
This matches the backtest logic exactly:
  - 1-day lag on macro signals (prevents forward-looking bias)
  - T+0 EMA confirmation (responsive to current market)
  - 28.1% rejection rate (filters bad entries)
  - Executes at open prices
        """
    )
    
    parser.add_argument('--port', type=int, default=4001,
                       help='IB port (4001=live Gateway, 4002=paper Gateway)')
    parser.add_argument('--live', action='store_true',
                       help='Execute live trades (default is dry run)')
    parser.add_argument('--no-telegram', action='store_true',
                       help='Disable Telegram notifications')
    
    args = parser.parse_args()
    
    # Initialize trader
    trader = SimpleLiveTrader(
        ib_port=args.port,
        dry_run=not args.live,
        enable_telegram=not args.no_telegram
    )
    
    # Run
    trader.run()


if __name__ == "__main__":
    main()


