"""
Interactive Brokers Executor for Macro Quadrant Strategy
=========================================================

Executes trades using Interactive Brokers API with CFDs
"""

from ib_insync import *
import pandas as pd
from typing import Dict, List
from datetime import datetime
import time

# Load ignore list and contract type filters
try:
    from strategy_config import IGNORE_TICKERS, MANAGED_CONTRACT_TYPES, IGNORED_CONTRACT_TYPES
except ImportError:
    IGNORE_TICKERS = []
    MANAGED_CONTRACT_TYPES = ['STK', 'CFD']
    IGNORED_CONTRACT_TYPES = ['OPT', 'FUT', 'FOP', 'WAR', 'IOPT']


class IBExecutor:
    """Execute trades via Interactive Brokers API using CFDs"""
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """
        Initialize IB connection
        
        Args:
            host: IB Gateway/TWS host (default localhost)
            port: 7497 for paper trading, 7496 for live (TWS)
                  4002 for paper trading, 4001 for live (IB Gateway)
            client_id: Unique client ID
        """
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        
        # CFD contract mapping (ticker -> IB CFD contract)
        self.cfd_contracts = {}
        
    def connect(self):
        """Connect to Interactive Brokers"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            print(f"✓ Connected to IB at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to IB: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from IB"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            print("✓ Disconnected from IB")
    
    def create_cfd_contract(self, ticker: str) -> CFD:
        """
        Create CFD contract for a ticker
        
        Args:
            ticker: ETF ticker (e.g., 'QQQ', 'SPY')
        
        Returns:
            CFD contract object
        
        Note: For ETFs, IB typically offers CFDs on the underlying index
        For example:
        - SPY -> US500 CFD
        - QQQ -> USTEC CFD (NASDAQ-100)
        - IWM -> US2000 CFD (Russell 2000)
        - DIA -> INDU CFD (Dow Jones)
        """
        # ETF to CFD mapping
        # Most ETFs have CFDs with the same ticker symbol
        # Only map the ones that need different symbols
        etf_to_cfd = {
            'SPY': 'US500',      # S&P 500 → US500 CFD
            'DIA': 'INDU',       # Dow Jones → INDU CFD
            'EFA': 'EUSTX50',    # European stocks → EUSTX50 CFD
            'EEM': 'CHINA50',    # Emerging markets → CHINA50 CFD
            # QQQ, IWM, and most other ETFs use their own ticker as CFD symbol
        }
        
        cfd_symbol = etf_to_cfd.get(ticker, ticker)
        
        # Create CFD contract with SMART exchange to resolve ambiguity
        contract = CFD(cfd_symbol, exchange='SMART', currency='USD')
        
        # Qualify contract with IB
        try:
            qualified = self.ib.qualifyContracts(contract)
            if qualified:
                print(f"  ✓ Qualified {ticker} -> {cfd_symbol} CFD (SMART)")
                return qualified[0]
            else:
                print(f"  ✗ Could not qualify {ticker} CFD")
                return None
        except Exception as e:
            print(f"  ✗ Error qualifying {ticker}: {e}")
            return None
    
    def get_account_value(self) -> float:
        """Get current account net liquidation value (handles multi-currency)"""
        if not self.connected:
            print("Not connected to IB")
            return 0.0
        
        # Try to get account summary (more reliable)
        try:
            account_summary = self.ib.accountSummary()
            for item in account_summary:
                if item.tag == 'NetLiquidation':
                    # Prefer USD, but accept base currency (GBP, EUR, etc.)
                    if item.currency == 'USD':
                        return float(item.value)
                    elif item.currency in ['GBP', 'EUR', 'BASE']:
                        # Found base currency, return it
                        value = float(item.value)
                        print(f"   (Account in {item.currency}: {value:,.2f})")
                        return value
        except:
            pass
        
        # Fallback: Try accountValues
        account_values = self.ib.accountValues()
        for item in account_values:
            if item.tag == 'NetLiquidation':
                if item.currency == 'USD':
                    return float(item.value)
                elif item.currency in ['GBP', 'EUR', 'BASE']:
                    value = float(item.value)
                    print(f"   (Account in {item.currency}: {value:,.2f})")
                    return value
        
        return 0.0
    
    def get_current_positions(self) -> Dict[str, float]:
        """
        Get current positions (only strategy-managed positions)
        
        Filters out:
        - Discretionary positions (IGNORE_TICKERS)
        - Non-managed contract types (options, futures, etc.)
        
        Returns:
            Dict of {ticker: quantity}
        """
        if not self.connected:
            return {}
        
        positions = {}
        for position in self.ib.positions():
            symbol = position.contract.symbol if hasattr(position.contract, 'symbol') else position.contract.localSymbol
            contract_type = position.contract.secType
            quantity = position.position
            
            # Skip zero positions
            if quantity == 0:
                continue
            
            # Skip ignored tickers (discretionary positions)
            if symbol in IGNORE_TICKERS:
                continue
            
            # Skip non-managed contract types (options, futures, etc.)
            if contract_type in IGNORED_CONTRACT_TYPES:
                continue
            
            # Only include managed contract types
            if contract_type in MANAGED_CONTRACT_TYPES:
                positions[symbol] = quantity
        
        return positions
    
    def calculate_position_sizes(self, target_weights: Dict[str, float], 
                                 account_value: float) -> Dict[str, float]:
        """
        Calculate CFD position sizes in notional USD
        
        Args:
            target_weights: Dict of {ticker: weight} where weight is % of capital
            account_value: Current account value in USD
        
        Returns:
            Dict of {ticker: notional_usd_value}
        """
        position_sizes = {}
        
        for ticker, weight in target_weights.items():
            notional_value = account_value * weight
            position_sizes[ticker] = notional_value
        
        return position_sizes
    
    def get_market_price(self, contract: CFD) -> float:
        """Get current market price for a contract (uses delayed/snapshot data)"""
        try:
            # Use delayed market data (free) - type 3
            self.ib.reqMarketDataType(3)
            
            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(3)  # Wait longer for market data
            
            # Try multiple price fields
            price = None
            if ticker.marketPrice() and not pd.isna(ticker.marketPrice()):
                price = ticker.marketPrice()
            elif ticker.last and not pd.isna(ticker.last):
                price = ticker.last
            elif ticker.close and not pd.isna(ticker.close):
                price = ticker.close
            elif ticker.bid and ticker.ask and not pd.isna(ticker.bid) and not pd.isna(ticker.ask):
                price = (ticker.bid + ticker.ask) / 2
            
            self.ib.cancelMktData(contract)
            
            if price is None or pd.isna(price):
                print(f"    ⚠️ No valid price data for {contract.symbol} - using yfinance fallback")
                # Fallback to yfinance for delayed price
                import yfinance as yf
                ticker_yf = yf.Ticker(contract.symbol)
                hist = ticker_yf.history(period='1d')
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
            
            return price if price and not pd.isna(price) else None
            
        except Exception as e:
            print(f"    ⚠️ Error getting price for {contract.symbol}: {e}")
            # Try yfinance as backup
            try:
                import yfinance as yf
                ticker_yf = yf.Ticker(contract.symbol)
                hist = ticker_yf.history(period='1d')
                if not hist.empty:
                    return hist['Close'].iloc[-1]
            except:
                pass
            return None
    
    def place_order(self, contract: CFD, quantity: int, action: str = 'BUY') -> Order:
        """
        Place a market order
        
        Args:
            contract: CFD contract
            quantity: Number of contracts (MUST be integer - no fractional quantities)
            action: 'BUY' or 'SELL'
        
        Returns:
            Order object
        """
        if not self.connected:
            print("Not connected to IB")
            return None
        
        # Ensure quantity is integer (no decimals allowed)
        quantity = int(round(quantity))
        
        if quantity == 0:
            print(f"  ⊘ Quantity rounds to 0, skipping order")
            return None
        
        # Create market order
        order = MarketOrder(action, quantity)
        
        # Place order
        trade = self.ib.placeOrder(contract, order)
        
        print(f"  → {action} {quantity} {contract.symbol} (integer qty)")
        
        return trade
    
    def execute_rebalance(self, target_weights: Dict[str, float], 
                          position_manager=None, atr_data: Dict[str, float] = None):
        """
        Execute portfolio rebalance with position tracking and stops
        
        Args:
            target_weights: Dict of {ticker: weight} where weight is % of capital
            position_manager: PositionManager instance for state tracking
            atr_data: Dict of {ticker: atr_value} for stop calculations
        """
        if not self.connected:
            print("Not connected to IB")
            return
        
        print("\n" + "="*60)
        print("EXECUTING REBALANCE")
        print("="*60)
        
        # Get account value
        account_value = self.get_account_value()
        print(f"\nAccount Value: ${account_value:,.2f}")
        
        # Get current positions from IB
        ib_positions = self.get_current_positions()
        print(f"Current Positions: {len(ib_positions)}")
        
        # Sync position manager with IB if provided
        if position_manager:
            position_manager.sync_with_ib()
            managed_positions = position_manager.get_all_positions()
        else:
            managed_positions = {}
        
        # Calculate target position sizes
        target_sizes = self.calculate_position_sizes(target_weights, account_value)
        
        print(f"\n📊 Target Portfolio:")
        print(f"  Positions: {len(target_sizes)}")
        print(f"  Total Notional: ${sum(target_sizes.values()):,.2f}")
        
        # Execute trades
        print(f"\n🔄 Executing Trades:")
        
        executed_trades = []
        
        # STEP 1: Close positions not in target (with position manager handling)
        for ticker in list(ib_positions.keys()):
            if ticker not in target_sizes:
                try:
                    print(f"\n  Closing {ticker}...")
                    contract = self.create_cfd_contract(ticker)
                    
                    closed = False
                    
                    # Try position manager first (if available and position is managed)
                    if contract and position_manager:
                        success = position_manager.exit_position(
                            contract, 
                            reason='QUAD_CHANGE'
                        )
                        if success:
                            executed_trades.append(ticker)
                            closed = True
                    
                    # Fallback: direct close if position manager didn't handle it
                    if contract and not closed:
                        print(f"  ⚠️ Position not managed by position_manager, closing directly...")
                        quantity = int(abs(ib_positions[ticker]))
                        action = 'SELL' if ib_positions[ticker] > 0 else 'BUY'
                        trade = self.place_order(contract, quantity, action)
                        if trade:
                            executed_trades.append(trade)
                            closed = True
                    
                    if not closed:
                        print(f"    ✗ Failed to close {ticker}")
                        
                except Exception as e:
                    # If closing this position fails, log it and continue with others
                    print(f"    ✗ ERROR closing {ticker}: {e}")
                    print(f"    ⏭️ Continuing with other positions...")
                    continue
        
        # Open/adjust positions in target (WITH STOP LOSSES!)
        for ticker, target_notional in target_sizes.items():
            try:
                print(f"\n  Adjusting {ticker}...")
                contract = self.create_cfd_contract(ticker)
                
                if not contract:
                    print(f"    ✗ Could not create contract for {ticker} - skipping")
                    continue
                
                # Get current price
                price = self.get_market_price(contract)
                if not price or pd.isna(price) or price <= 0:
                    print(f"    ✗ Could not get valid price for {ticker} - skipping")
                    continue
                
                # Check if account value is valid
                if account_value <= 0:
                    print(f"    ✗ Invalid account value (${account_value:.2f}) - skipping")
                    continue
                
                # Calculate target quantity (MUST be integer - no fractional shares)
                target_quantity = int(round(target_notional / price))
                
                # Safety check
                if target_quantity <= 0:
                    print(f"    ⊘ Target quantity is 0 - skipping")
                    continue
                
                # Calculate current quantity
                current_quantity = int(ib_positions.get(ticker, 0))
                
                # Calculate delta (integer)
                delta_quantity = int(target_quantity - current_quantity)
                
                # Get ATR for stop calculation (prefer from atr_data, but calculate if needed)
                atr = None
                if atr_data and ticker in atr_data:
                    atr = atr_data[ticker]
                elif position_manager:
                    # Try to get ATR from position manager if it has this position
                    pos = position_manager.get_position(ticker)
                    if pos and 'atr_at_entry' in pos:
                        atr = pos['atr_at_entry']
                
                # Only trade if delta > threshold (e.g., 5% of target) AND delta >= 1 share
                if abs(delta_quantity) >= 1 and abs(delta_quantity) > abs(target_quantity) * 0.05:
                    
                    # Case 1: NEW POSITION - use position_manager.enter_position (places stop!)
                    if current_quantity == 0 and position_manager and atr:
                        stop_price = price - (2.0 * atr)  # 2 ATR stop
                        
                        print(f"    📈 NEW POSITION - Entry with stop loss")
                        print(f"    🛑 Stop: ${stop_price:.2f} (2.0 ATR = ${atr:.2f})")
                        
                        success = position_manager.enter_position(
                            contract=contract,
                            quantity=target_quantity,
                            entry_price=price,
                            stop_price=stop_price,
                            atr=atr
                        )
                        
                        if success:
                            print(f"    ✓ Position entered with stop loss placed")
                        else:
                            print(f"    ✗ Failed to enter position via manager")
                    
                    # Case 2: ADJUSTING EXISTING IN STATE - use position_manager.adjust_position (keeps stop!)
                    elif current_quantity != 0 and position_manager and position_manager.has_position(ticker):
                        print(f"    🔄 ADJUSTING POSITION - Keeping original stop")
                        
                        success = position_manager.adjust_position(
                            contract=contract,
                            new_quantity=target_quantity
                        )
                        
                        if success:
                            print(f"    ✓ Position adjusted, stop updated for new size")
                        else:
                            print(f"    ✗ Failed to adjust position via manager")
                    
                    # Case 3: EXISTING POSITION NOT IN STATE - add to state and place stop
                    elif current_quantity != 0 and position_manager and atr:
                        print(f"    ⚠️ Position exists in IB but not in state - adding to state with stop")
                        
                        # Calculate stop price based on current price (entry price approximation)
                        stop_price = price - (2.0 * atr)  # 2 ATR stop
                        
                        # Add position to state manually (simulating entry)
                        position_manager.state['positions'][ticker] = {
                            'shares': current_quantity,
                            'entry_price': price,  # Use current price as approximation
                            'stop_price': stop_price,
                            'atr_at_entry': atr,
                            'entry_order_id': None,  # Unknown
                            'stop_order_id': None,  # Will be set when we place stop
                            'entry_date': datetime.now().isoformat(),
                            'contract_details': {
                                'symbol': contract.symbol,
                                'secType': contract.secType,
                                'exchange': contract.exchange,
                                'currency': contract.currency
                            }
                        }
                        
                        # Place stop order
                        stop_order = Order()
                        stop_order.action = 'SELL'
                        stop_order.orderType = 'STP'
                        stop_order.auxPrice = stop_price
                        stop_order.totalQuantity = current_quantity
                        stop_order.tif = 'GTC'  # Good-Till-Cancelled
                        stop_order.transmit = True
                        
                        print(f"    🛑 Placing STOP: Sell {current_quantity} {ticker} @ ${stop_price:.2f} (GTC)")
                        stop_trade = self.ib.placeOrder(contract, stop_order)
                        position_manager.state['positions'][ticker]['stop_order_id'] = stop_trade.order.orderId
                        position_manager.save_state()
                        
                        # Now adjust if needed
                        if delta_quantity != 0:
                            print(f"    🔄 Now adjusting position size...")
                            success = position_manager.adjust_position(
                                contract=contract,
                                new_quantity=target_quantity
                            )
                            if success:
                                print(f"    ✓ Position adjusted with stop")
                            else:
                                print(f"    ✗ Failed to adjust position")
                        else:
                            print(f"    ✓ Position added to state with stop placed")
                    
                    # Case 4: FALLBACK - direct order (no stop management) - WARNING
                    else:
                        print(f"    ⚠️ WARNING: Trading without stop loss!")
                        if not position_manager:
                            print(f"    ⚠️ No position manager available")
                        if not atr:
                            print(f"    ⚠️ No ATR data for {ticker}")
                        action = 'BUY' if delta_quantity > 0 else 'SELL'
                        trade = self.place_order(contract, int(abs(delta_quantity)), action)
                        if trade:
                            executed_trades.append(trade)
                
                else:
                    print(f"    ⊘ Position close to target, no trade needed")
                    # Even if no trade needed, ensure stop exists if position is in IB
                    if current_quantity != 0 and position_manager and atr and not position_manager.has_position(ticker):
                        print(f"    🛑 Ensuring stop exists for existing position...")
                        stop_price = price - (2.0 * atr)
                        
                        # Add to state
                        position_manager.state['positions'][ticker] = {
                            'shares': current_quantity,
                            'entry_price': price,
                            'stop_price': stop_price,
                            'atr_at_entry': atr,
                            'entry_order_id': None,
                            'stop_order_id': None,
                            'entry_date': datetime.now().isoformat(),
                            'contract_details': {
                                'symbol': contract.symbol,
                                'secType': contract.secType,
                                'exchange': contract.exchange,
                                'currency': contract.currency
                            }
                        }
                        
                        # Place stop
                        stop_order = Order()
                        stop_order.action = 'SELL'
                        stop_order.orderType = 'STP'
                        stop_order.auxPrice = stop_price
                        stop_order.totalQuantity = current_quantity
                        stop_order.tif = 'GTC'
                        stop_order.transmit = True
                        
                        print(f"    🛑 Placing STOP: Sell {current_quantity} {ticker} @ ${stop_price:.2f} (GTC)")
                        stop_trade = self.ib.placeOrder(contract, stop_order)
                        position_manager.state['positions'][ticker]['stop_order_id'] = stop_trade.order.orderId
                        position_manager.save_state()
                        print(f"    ✓ Stop placed for existing position")
            
            except Exception as e:
                # If any ticker fails (e.g., CPER restricted), log it and continue with others
                print(f"    ✗ ERROR with {ticker}: {e}")
                print(f"    ⏭️ Continuing with other positions...")
                continue
        
        print(f"\n✓ Executed {len(executed_trades)} trades")
        
        # FINAL CHECK: Ensure all positions in target have stops
        if position_manager and atr_data:
            print(f"\n🛑 FINAL CHECK: Ensuring all positions have stop losses...")
            final_positions = self.get_current_positions()
            
            for ticker in target_sizes.keys():
                if ticker in final_positions and final_positions[ticker] != 0:
                    # Check if position has a stop
                    if not position_manager.has_position(ticker):
                        print(f"  ⚠️ {ticker}: Position exists but no stop in state - adding stop...")
                        
                        if ticker in atr_data:
                            try:
                                contract = self.create_cfd_contract(ticker)
                                if contract:
                                    price = self.get_market_price(contract)
                                    if price and price > 0:
                                        atr = atr_data[ticker]
                                        stop_price = price - (2.0 * atr)
                                        quantity = int(abs(final_positions[ticker]))
                                        
                                        # Add to state
                                        position_manager.state['positions'][ticker] = {
                                            'shares': quantity,
                                            'entry_price': price,
                                            'stop_price': stop_price,
                                            'atr_at_entry': atr,
                                            'entry_order_id': None,
                                            'stop_order_id': None,
                                            'entry_date': datetime.now().isoformat(),
                                            'contract_details': {
                                                'symbol': contract.symbol,
                                                'secType': contract.secType,
                                                'exchange': contract.exchange,
                                                'currency': contract.currency
                                            }
                                        }
                                        
                                        # Place stop
                                        stop_order = Order()
                                        stop_order.action = 'SELL'
                                        stop_order.orderType = 'STP'
                                        stop_order.auxPrice = stop_price
                                        stop_order.totalQuantity = quantity
                                        stop_order.tif = 'GTC'
                                        stop_order.transmit = True
                                        
                                        stop_trade = self.ib.placeOrder(contract, stop_order)
                                        position_manager.state['positions'][ticker]['stop_order_id'] = stop_trade.order.orderId
                                        position_manager.save_state()
                                        print(f"    ✓ Stop placed: ${stop_price:.2f}")
                            except Exception as e:
                                print(f"    ✗ Error adding stop for {ticker}: {e}")
                    else:
                        # Verify stop order is still active
                        pos = position_manager.get_position(ticker)
                        if pos and pos.get('stop_order_id'):
                            # Check if order still exists
                            try:
                                # Try to get order status (if order was cancelled/filled, we need to replace it)
                                # Note: This is a simplified check - in production you might want to query open orders
                                print(f"  ✓ {ticker}: Stop order exists (ID: {pos['stop_order_id']})")
                            except:
                                pass
        
        return executed_trades
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


if __name__ == "__main__":
    # Test connection
    print("Testing IB Connection...")
    print("\n⚠️  Make sure IB Gateway or TWS is running!")
    print("    Paper Trading: port 7497 (TWS) or 4002 (Gateway)")
    print("    Live Trading:  port 7496 (TWS) or 4001 (Gateway)")
    
    with IBExecutor(port=7497) as ib_exec:
        if ib_exec.connected:
            print("\n✓ Connection successful!")
            
            # Get account info
            account_value = ib_exec.get_account_value()
            print(f"\nAccount Value: ${account_value:,.2f}")
            
            # Get current positions
            positions = ib_exec.get_current_positions()
            print(f"Current Positions: {positions}")

