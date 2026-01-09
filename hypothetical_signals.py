"""
Hypothetical Signal Generator
==============================

Generates theoretical trading signals and sends them to Telegram.
No actual trades are executed - this is for analysis only.
"""

from signal_generator import SignalGenerator
from telegram_notifier import get_notifier
from datetime import datetime


def format_telegram_message(signals: dict) -> str:
    """Format signals as a Telegram message - simplified version"""
    top1, top2 = signals['top_quadrants']
    target_weights = signals['target_weights']
    excluded_below_ema = signals.get('excluded_below_ema', {})
    timestamp = signals.get('timestamp', datetime.now())
    price_date = signals.get('price_date', timestamp.date())
    analysis_timestamp_utc = signals.get('analysis_timestamp_utc', datetime.utcnow())
    
    # Build message - only primary, secondary, and positions
    message = f"{price_date}\n\n"
    message += f"<b>Price Data Date:</b> {price_date}\n"
    message += f"<b>Analysis Time (UTC):</b> {analysis_timestamp_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
    message += f"<b>Primary Quad:</b> {top1}\n"
    message += f"<b>Secondary Quad:</b> {top2}\n\n"
    
    if not target_weights:
        message += "<b>Positions:</b> None (100% Cash)\n"
    else:
        # Sort by weight
        sorted_weights = sorted(target_weights.items(), key=lambda x: x[1], reverse=True)
        
        message += "<b>Positions:</b>\n"
        for ticker, weight in sorted_weights:
            message += f"{ticker}: {weight*100:.2f}%\n"
    
    # Add excluded assets section if any exist
    if excluded_below_ema:
        message += "\n<b>Excluded (Below EMA):</b>\n"
        # Sort by would-be weight for consistency
        sorted_excluded = sorted(excluded_below_ema.items(), 
                                key=lambda x: x[1].get('would_be_weight', 0), 
                                reverse=True)
        for ticker, info in sorted_excluded:
            price = info.get('price', 0)
            ema_val = info.get('ema', 0)
            pct_below = ((price - ema_val) / ema_val * 100) if ema_val > 0 else 0
            message += f"{ticker}: {pct_below:.2f}% below EMA\n"
    
    return message


def main():
    """Generate hypothetical signals and send to Telegram"""
    print("=" * 70)
    print("HYPOTHETICAL SIGNAL GENERATOR")
    print("=" * 70)
    print("\nGenerating theoretical trading signals...")
    print("(No actual trades will be executed)\n")
    
    try:
        # Initialize signal generator
        sg = SignalGenerator(
            momentum_days=20,
            ema_period=50,
            vol_lookback=30,
            max_positions=10,
            atr_stop_loss=2.0,
            atr_period=14
        )
        
        # Generate signals
        signals = sg.generate_signals()
        
        # Format message
        message = format_telegram_message(signals)
        
        # Send to Telegram
        print("\n" + "=" * 70)
        print("SENDING TO TELEGRAM")
        print("=" * 70)
        
        notifier = get_notifier()
        success = notifier.send_message(message)
        
        if success:
            print("\n✅ Successfully sent hypothetical signals to Telegram!")
            print("\nMessage preview:")
            print("-" * 70)
            # Print a plain text version for console
            print(message.replace('<b>', '').replace('</b>', '')
                  .replace('<i>', '').replace('</i>', ''))
        else:
            print("\n❌ Failed to send message to Telegram")
            print("Check your Telegram token and chat ID")
        
    except Exception as e:
        print(f"\n❌ Error generating signals: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to send error to Telegram
        try:
            notifier = get_notifier()
            notifier.send_error_alert(str(e), "Hypothetical Signal Generator")
        except:
            pass


if __name__ == "__main__":
    main()

