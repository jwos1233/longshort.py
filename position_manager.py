"""
Position Manager for ATR Stop Loss Tracking
============================================

Manages position state, entry prices, stop orders, and trade history.
Critical for the ATR 2.0x stop loss strategy.
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional, List
from ib_insync import IB, Contract, Order, Trade
import pandas as pd


class PositionManager:
    """
    Manages position state and stop orders for live trading
    
    Features:
    - Tracks entry prices and stop levels
    - Places and manages IB stop orders
    - Persists state to JSON file
    - Logs all trades with entry/exit reasons
    - Syncs with IB positions on startup
    """
    
    def __init__(self, ib: IB, state_file='position_state.json', 
                 trade_log='trade_history.csv', account: str | None = None):
        """
        Initialize Position Manager
        
        Args:
            ib: Connected IB instance
            state_file: Path to JSON state file
            trade_log: Path to CSV trade log
        """
        self.ib = ib
        self.state_file = state_file
        self.trade_log = trade_log
        self.account = (account or "").strip()
        self.state = self.load_state()
        self.pending_orders = {}  # Track orders being placed
        
    def load_state(self) -> Dict:
        """Load position state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading state file: {e}")
                return {'positions': {}, 'metadata': {}}
        else:
            return {'positions': {}, 'metadata': {}}
    
    def save_state(self):
        """Save position state to file"""
        try:
            self.state['metadata']['last_updated'] = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving state file: {e}")
    
    def sync_with_ib(self):
        """
        Sync local state with IB positions
        Handles cases where positions were closed outside the system
        """
        acct = self.account if getattr(self, "account", None) else ""
        ib_positions_list = self.ib.positions(acct) if acct else self.ib.positions()
        ib_positions = {p.contract.symbol: p.position for p in ib_positions_list}
        
        # Check for positions that exist in state but not in IB (were closed)
        closed_tickers = []
        for ticker in list(self.state['positions'].keys()):
            if ticker not in ib_positions or ib_positions[ticker] == 0:
                print(f"  ⚠️ {ticker} no longer in IB, removing from state")
                closed_tickers.append(ticker)
        
        # Clean up closed positions
        for ticker in closed_tickers:
            self._cleanup_position(ticker, 'EXTERNAL_CLOSE')
        
        # Check for positions in IB but not in state (were opened externally)
        for ticker, quantity in ib_positions.items():
            if quantity != 0 and ticker not in self.state['positions']:
                print(f"  ⚠️ {ticker} found in IB but not in state (external position)")
                # Don't manage external positions
        
        self.save_state()
        return len(closed_tickers)
    
    def enter_position(self, contract: Contract, quantity: int, 
                      entry_price: float, stop_price: float, 
                      atr: float) -> bool:
        """
        Enter a position and place stop order
        
        Args:
            contract: IB contract
            quantity: Number of shares/contracts
            entry_price: Entry price (for reference)
            stop_price: Stop loss price
            atr: ATR value at entry
            
        Returns:
            True if successful, False otherwise
        """
        ticker = contract.symbol
        
        try:
            # 1. Place entry order (market order)
            entry_order = Order()
            entry_order.action = 'BUY'
            entry_order.orderType = 'MKT'
            entry_order.totalQuantity = quantity
            entry_order.transmit = True  # Send immediately
            if getattr(self, "account", None):
                entry_order.account = self.account
            
            print(f"  📈 Placing BUY order: {quantity} {ticker} @ market")
            entry_trade = self.ib.placeOrder(contract, entry_order)
            
            # Wait for fill (with timeout)
            for i in range(30):  # 30 seconds max
                self.ib.sleep(1)
                if entry_trade.orderStatus.status in ['Filled', 'Cancelled']:
                    break
            
            if entry_trade.orderStatus.status != 'Filled':
                print(f"  ✗ Entry order not filled: {entry_trade.orderStatus.status}")
                return False
            
            # Get actual fill price
            actual_entry_price = entry_trade.orderStatus.avgFillPrice
            print(f"  ✓ Filled: {quantity} {ticker} @ ${actual_entry_price:.2f}")
            
            # 2. Place stop order (GTC = Good-Till-Cancelled)
            stop_order = Order()
            stop_order.action = 'SELL'
            stop_order.orderType = 'STP'
            stop_order.auxPrice = stop_price
            stop_order.totalQuantity = quantity
            stop_order.tif = 'GTC'  # Good-Till-Cancelled (doesn't expire daily)
            stop_order.transmit = True
            if getattr(self, "account", None):
                stop_order.account = self.account
            
            print(f"  🛑 Placing STOP: Sell {quantity} {ticker} @ ${stop_price:.2f} (GTC)")
            stop_trade = self.ib.placeOrder(contract, stop_order)
            
            # 3. Save state
            self.state['positions'][ticker] = {
                'shares': quantity,
                'entry_price': actual_entry_price,
                'stop_price': stop_price,
                'atr_at_entry': atr,
                'entry_order_id': entry_trade.order.orderId,
                'stop_order_id': stop_trade.order.orderId,
                'entry_date': datetime.now().isoformat(),
                'contract_details': {
                    'symbol': contract.symbol,
                    'secType': contract.secType,
                    'exchange': contract.exchange,
                    'currency': contract.currency
                }
            }
            self.save_state()
            
            # 4. Log trade
            self._log_trade({
                'date': datetime.now().isoformat(),
                'ticker': ticker,
                'action': 'ENTRY',
                'quantity': quantity,
                'price': actual_entry_price,
                'stop_price': stop_price,
                'atr': atr,
                'reason': 'NEW_SIGNAL'
            })
            
            print(f"  ✓ Position opened: {ticker}")
            return True
            
        except Exception as e:
            print(f"  ✗ Error entering position {ticker}: {e}")
            return False
    
    def adjust_position(self, contract: Contract, new_quantity: int) -> bool:
        """
        Adjust an existing position size (CRITICAL: keeps original stop price)
        
        This is used when rebalancing changes position size.
        The stop price is NEVER moved - only the quantity is adjusted.
        This matches the backtest behavior.
        
        Args:
            contract: IB contract
            new_quantity: New total quantity desired
            
        Returns:
            True if successful, False otherwise
        """
        ticker = contract.symbol
        
        if ticker not in self.state['positions']:
            print(f"  ⚠️ Cannot adjust {ticker} - no existing position")
            return False
        
        position = self.state['positions'][ticker]
        old_quantity = position['shares']
        
        # No change needed
        if new_quantity == old_quantity:
            print(f"  ℹ️ {ticker}: No adjustment needed ({old_quantity} shares)")
            return True
        
        try:
            # 1. Cancel old stop order
            old_stop_order_id = position.get('stop_order_id')
            if old_stop_order_id:
                try:
                    self.ib.cancelOrder(old_stop_order_id)
                    print(f"  ✓ Cancelled old stop order for {ticker}")
                except Exception as e:
                    print(f"  ⚠️ Could not cancel old stop: {e}")
            
            # 2. Place adjustment order (buy more or sell some)
            delta = new_quantity - old_quantity
            action = 'BUY' if delta > 0 else 'SELL'
            abs_delta = abs(delta)
            
            adjustment_order = Order()
            adjustment_order.action = action
            adjustment_order.orderType = 'MKT'
            adjustment_order.totalQuantity = abs_delta
            adjustment_order.transmit = True
            if getattr(self, "account", None):
                adjustment_order.account = self.account
            
            print(f"  🔄 Adjusting {ticker}: {old_quantity} → {new_quantity} ({action} {abs_delta})")
            adjust_trade = self.ib.placeOrder(contract, adjustment_order)
            
            # Wait for fill
            for i in range(30):
                self.ib.sleep(1)
                if adjust_trade.orderStatus.status in ['Filled', 'Cancelled']:
                    break
            
            if adjust_trade.orderStatus.status != 'Filled':
                print(f"  ✗ Adjustment order not filled: {adjust_trade.orderStatus.status}")
                return False
            
            fill_price = adjust_trade.orderStatus.avgFillPrice
            print(f"  ✓ Filled: {action} {abs_delta} {ticker} @ ${fill_price:.2f}")
            
            # 3. Place NEW stop order at ORIGINAL stop price with NEW quantity
            # THIS IS CRITICAL: We use the ORIGINAL stop_price, NOT a new calculation
            original_stop_price = position['stop_price']
            
            new_stop_order = Order()
            new_stop_order.action = 'SELL'
            new_stop_order.orderType = 'STP'
            new_stop_order.auxPrice = original_stop_price  # SAME stop price!
            new_stop_order.totalQuantity = new_quantity
            new_stop_order.tif = 'GTC'  # Good-Till-Cancelled (doesn't expire daily)
            new_stop_order.transmit = True
            if getattr(self, "account", None):
                new_stop_order.account = self.account
            
            print(f"  🛑 New STOP: Sell {new_quantity} {ticker} @ ${original_stop_price:.2f} (GTC, SAME price)")
            new_stop_trade = self.ib.placeOrder(contract, new_stop_order)
            
            # 4. Update state with new quantity and stop order ID
            # KEEP original entry_price and stop_price!
            position['shares'] = new_quantity
            position['stop_order_id'] = new_stop_trade.order.orderId
            position['last_adjusted'] = datetime.now().isoformat()
            self.save_state()
            
            # 5. Log adjustment
            self._log_trade({
                'date': datetime.now().isoformat(),
                'ticker': ticker,
                'action': 'ADJUST',
                'old_quantity': old_quantity,
                'new_quantity': new_quantity,
                'delta': delta,
                'fill_price': fill_price,
                'stop_price': original_stop_price,  # Log that stop didn't move
                'reason': 'REBALANCE'
            })
            
            print(f"  ✓ Position adjusted: {ticker} ({old_quantity} → {new_quantity})")
            print(f"  🛑 Stop remains at: ${original_stop_price:.2f} (UNCHANGED)")
            return True
            
        except Exception as e:
            print(f"  ✗ Error adjusting position {ticker}: {e}")
            return False
    
    def exit_position(self, contract: Contract, reason: str, 
                     current_price: Optional[float] = None) -> bool:
        """
        Exit a position and cancel stop order
        
        Args:
            contract: IB contract
            reason: Exit reason (EMA_CROSS, QUAD_CHANGE, TOP10_DROP, ATR_STOP)
            current_price: Current market price (optional)
            
        Returns:
            True if successful, False otherwise
        """
        ticker = contract.symbol
        
        if ticker not in self.state['positions']:
            print(f"  ⚠️ {ticker} not in state, cannot exit")
            return False
        
        position = self.state['positions'][ticker]
        
        try:
            # 1. Cancel stop order first
            stop_order_id = position.get('stop_order_id')
            if stop_order_id:
                try:
                    self.ib.cancelOrder(stop_order_id)
                    print(f"  ✓ Cancelled stop order for {ticker}")
                except Exception as e:
                    print(f"  ⚠️ Could not cancel stop for {ticker}: {e}")
                    # Continue anyway - might already be cancelled/filled
            
            # 2. Place exit order
            exit_order = Order()
            exit_order.action = 'SELL'
            exit_order.orderType = 'MKT'
            exit_order.totalQuantity = position['shares']
            exit_order.transmit = True
            if getattr(self, "account", None):
                exit_order.account = self.account
            
            print(f"  📉 Placing SELL order: {position['shares']} {ticker} @ market (reason: {reason})")
            exit_trade = self.ib.placeOrder(contract, exit_order)
            
            # Wait for fill
            for i in range(30):
                self.ib.sleep(1)
                if exit_trade.orderStatus.status in ['Filled', 'Cancelled']:
                    break
            
            if exit_trade.orderStatus.status != 'Filled':
                print(f"  ✗ Exit order not filled: {exit_trade.orderStatus.status}")
                return False
            
            # Get actual exit price
            actual_exit_price = exit_trade.orderStatus.avgFillPrice
            print(f"  ✓ Filled: Sold {position['shares']} {ticker} @ ${actual_exit_price:.2f}")
            
            # 3. Calculate P&L
            pnl = (actual_exit_price - position['entry_price']) * position['shares']
            pnl_pct = (actual_exit_price / position['entry_price'] - 1) * 100
            
            print(f"  💰 P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            
            # 4. Log trade
            self._log_trade({
                'date': datetime.now().isoformat(),
                'ticker': ticker,
                'action': 'EXIT',
                'quantity': position['shares'],
                'entry_price': position['entry_price'],
                'exit_price': actual_exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'reason': reason,
                'days_held': (datetime.now() - datetime.fromisoformat(position['entry_date'])).days
            })
            
            # 5. Remove from state
            del self.state['positions'][ticker]
            self.save_state()
            
            print(f"  ✓ Position closed: {ticker}")
            return True
            
        except Exception as e:
            print(f"  ✗ Error exiting position {ticker}: {e}")
            return False
    
    def _cleanup_position(self, ticker: str, reason: str):
        """Clean up position from state (used for external closes)"""
        if ticker in self.state['positions']:
            position = self.state['positions'][ticker]
            
            # Try to cancel stop if it exists
            stop_order_id = position.get('stop_order_id')
            if stop_order_id:
                try:
                    self.ib.cancelOrder(stop_order_id)
                except:
                    pass
            
            # Log cleanup
            self._log_trade({
                'date': datetime.now().isoformat(),
                'ticker': ticker,
                'action': 'CLEANUP',
                'quantity': position['shares'],
                'entry_price': position['entry_price'],
                'reason': reason
            })
            
            # Remove from state
            del self.state['positions'][ticker]
    
    def check_stops(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Check if any stops should be hit based on current prices
        Note: IB will execute stops automatically, this is for monitoring
        
        Args:
            current_prices: Dict of {ticker: current_price}
            
        Returns:
            List of tickers where stop might be hit
        """
        stops_hit = []
        
        for ticker, position in self.state['positions'].items():
            if ticker in current_prices:
                current_price = current_prices[ticker]
                stop_price = position['stop_price']
                
                if current_price <= stop_price:
                    print(f"  ⚠️ {ticker}: Price ${current_price:.2f} <= Stop ${stop_price:.2f}")
                    stops_hit.append(ticker)
        
        return stops_hit
    
    def get_position(self, ticker: str) -> Optional[Dict]:
        """Get position details for a ticker"""
        return self.state['positions'].get(ticker)
    
    def get_all_positions(self) -> Dict:
        """Get all current positions"""
        return self.state['positions'].copy()
    
    def has_position(self, ticker: str) -> bool:
        """Check if we have a position in ticker"""
        return ticker in self.state['positions']
    
    def _log_trade(self, trade_data: Dict):
        """Log trade to CSV file"""
        try:
            # Create DataFrame
            df = pd.DataFrame([trade_data])
            
            # Append to CSV
            if os.path.exists(self.trade_log):
                df.to_csv(self.trade_log, mode='a', header=False, index=False)
            else:
                df.to_csv(self.trade_log, mode='w', header=True, index=False)
                
        except Exception as e:
            print(f"⚠️ Error logging trade: {e}")
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if os.path.exists(self.trade_log):
            return pd.read_csv(self.trade_log)
        else:
            return pd.DataFrame()
    
    def print_summary(self):
        """Print current position summary"""
        positions = self.state['positions']
        
        if not positions:
            print("No open positions")
            return
        
        print(f"\n{'='*70}")
        print(f"OPEN POSITIONS ({len(positions)})")
        print(f"{'='*70}")
        print(f"{'Ticker':<8} {'Shares':>8} {'Entry':>10} {'Stop':>10} {'Days':>6}")
        print(f"{'-'*70}")
        
        for ticker, pos in positions.items():
            days_held = (datetime.now() - datetime.fromisoformat(pos['entry_date'])).days
            print(f"{ticker:<8} {pos['shares']:>8} ${pos['entry_price']:>9.2f} "
                  f"${pos['stop_price']:>9.2f} {days_held:>6}")
        
        print(f"{'='*70}\n")
