##Telegram Notification System
##============================

##Send alerts for night and morning trading runs
##"""

import requests
from datetime import datetime
from typing import Dict, List


class TelegramNotifier:
    """Send trading alerts via Telegram"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_message(self, message: str):
        """Send a message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                print(f"Telegram API error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")
            return False
    
    def send_night_alert(self, signals: Dict, pending_count: int, 
                         account_value: float, current_positions: Dict = None):
        """Send night run summary with USD values
        
        Args:
            account_value: REAL account value from IB Gateway (required, no default)
        """
        regime = signals.get('current_regime', 'Unknown')
        top_quads = signals.get('top_quadrants', ('?', '?'))
        leverage = signals.get('total_leverage', 0)
        
        # Get price date and UTC timestamp if available
        price_date = signals.get('price_date', None)
        analysis_timestamp_utc = signals.get('analysis_timestamp_utc', None)
        
        # Get all positions by weight
        weights = signals.get('target_weights', {})
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate total target value
        total_target = sum(account_value * w for w in weights.values())
        
        message = f"""
<b>🌙 NIGHT SIGNAL GENERATION</b>
{datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        # Add price date and UTC timestamp if available
        if price_date:
            message += f"<b>Price Data Date:</b> {price_date}\n"
        if analysis_timestamp_utc:
            if isinstance(analysis_timestamp_utc, datetime):
                message += f"<b>Analysis Time (UTC):</b> {analysis_timestamp_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        
        message += f"""
<b>Market Regime:</b> {regime}
<b>Top Quadrants:</b> {top_quads[0]}, {top_quads[1]}
<b>Target Leverage:</b> {leverage:.2f}x
<b>Account Value:</b> ${account_value:,.2f} 💰
"""
        
        # Show current positions if available
        if current_positions and len(current_positions) > 0:
            message += f"\n<b>📊 Current Positions: {len(current_positions)}</b>\n"
            for ticker in sorted(current_positions.keys())[:5]:
                message += f"  • {ticker}\n"
            if len(current_positions) > 5:
                message += f"  <i>...and {len(current_positions) - 5} more</i>\n"
        
        # Show all target positions with USD
        message += f"\n<b>🎯 Target Positions: {len(sorted_weights)}</b>\n"
        message += f"<b>Total Target: ${total_target:,.0f}</b>\n\n"
        
        for ticker, weight in sorted_weights:
            target_usd = account_value * weight
            message += f"  • {ticker}: {weight*100:.1f}% (${target_usd:,.0f})\n"
        
        # Expected rejections
        expected_rejections = int(len(sorted_weights) * 0.28)
        expected_confirmed = len(sorted_weights) - expected_rejections
        
        message += f"""
<b>⏰ Tomorrow Morning (9:29 AM):</b>
Expected confirmed: ~{expected_confirmed}
Expected rejected: ~{expected_rejections} (28%)

<i>Full details: night_plan_{datetime.now().strftime('%Y%m%d')}.txt</i>
"""
        
        self.send_message(message)
    
    def send_morning_alert(self, confirmed: Dict, rejected: Dict, 
                          trades_executed: List, positions_summary: Dict):
        """Send morning execution summary"""
        
        total = len(confirmed) + len(rejected)
        rejection_rate = (len(rejected) / total * 100) if total > 0 else 0
        
        message = f"""
<b>☀️ MORNING EXECUTION COMPLETE</b>
{datetime.now().strftime('%Y-%m-%d %H:%M')}

<b>CONFIRMATION RESULTS:</b>
✅ Confirmed: {len(confirmed)}
❌ Rejected: {len(rejected)}
📊 Rejection Rate: {rejection_rate:.1f}%
"""
        
        if confirmed:
            message += "\n<b>Confirmed Entries:</b>\n"
            for ticker in sorted(confirmed.keys()):
                message += f"  ✅ {ticker}\n"
        
        if rejected:
            message += "\n<b>Rejected Entries:</b>\n"
            for ticker in sorted(rejected.keys()):
                message += f"  ❌ {ticker}\n"
        
        # Position changes
        added = positions_summary.get('added', [])
        removed = positions_summary.get('removed', [])
        adjusted = positions_summary.get('adjusted', [])
        
        if added or removed or adjusted:
            message += "\n<b>POSITION CHANGES:</b>\n"
            
            if added:
                message += f"  📈 New: {len(added)}\n"
                for item in added[:3]:  # Show first 3
                    message += f"    • {item}\n"
            
            if removed:
                message += f"  📉 Closed: {len(removed)}\n"
                for item in removed[:3]:
                    message += f"    • {item}\n"
            
            if adjusted:
                message += f"  🔄 Adjusted: {len(adjusted)}\n"
        
        # Trades
        trade_count = len(trades_executed) if trades_executed else 0
        message += f"\n<b>💼 Trades Executed: {trade_count}</b>\n"
        
        if trade_count == 0:
            message += "<i>No changes needed (positions within 5% threshold)</i>\n"
        
        message += f"\n<i>Full report: morning_report_{datetime.now().strftime('%Y%m%d')}.txt</i>"
        
        self.send_message(message)
    
    def send_error_alert(self, error_message: str, context: str = "Unknown"):
        """Send error alert"""
        message = f"""
<b>🚨 ERROR ALERT</b>
{datetime.now().strftime('%Y-%m-%d %H:%M')}

<b>Context:</b> {context}

<b>Error:</b>
<code>{error_message}</code>

<i>Check logs for details</i>
"""
        self.send_message(message)
    
    def send_test_message(self):
        """Send test message to verify connection"""
        message = f"""
<b>✅ TEST MESSAGE</b>
{datetime.now().strftime('%Y-%m-%d %H:%M')}

Telegram notifications are working!
You will receive alerts for:
  • Night signal generation
  • Morning execution results
  • Errors and warnings

<i>Macro Quadrant Strategy - Live Trading</i>
"""
        return self.send_message(message)


# Initialize with your credentials
TELEGRAM_TOKEN = "7206335521:AAGQeuhik1SrN_qMakb9bxkI1iAJmg8A3Wo"
TELEGRAM_CHAT_ID = "7119645510"


def get_notifier():
    """Get configured Telegram notifier"""
    return TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)


if __name__ == "__main__":
    # Test Telegram connection
    print("Testing Telegram connection...")
    notifier = get_notifier()
    
    if notifier.send_test_message():
        print("[SUCCESS] Test message sent successfully!")
        print("Check your Telegram for the message")
    else:
        print("[FAILED] Failed to send test message")
        print("Check your token and chat ID")
