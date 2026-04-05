# main.py - 27 STOCKS VERSION
#  run this : streamlit run web_app.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
from datetime import datetime
import yfinance as yf
import pandas as pd

def quick_stock_analysis():
    print("🚀 AI STOCK PREDICTOR - LIVE (27 STOCKS)")
    print("=" * 80)
    
    # ALL 27 STOCKS
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'AMD', 'INTC', 
               'JPM', 'BAC', 'GS', 'MS', 'JNJ', 'PFE', 'UNH', 'MRK', 'XOM', 'CVX', 'COP', 
               'WMT', 'TGT', 'COST', 'SPY', 'QQQ', 'DIA']
    
    while True:
        print(f"\n📊 LIVE UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Display in groups for better readability
        group_size = 9
        for i in range(0, len(symbols), group_size):
            group = symbols[i:i + group_size]
            print("\n" + "-" * 80)
            
            for symbol in group:
                try:
                    # Get real-time data
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period="2d")
                    
                    if len(hist) >= 2:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2]
                        change = ((current_price - prev_price) / prev_price) * 100
                        
                        # Simple prediction logic
                        if change > 0:
                            prediction = "📈 BULLISH"
                            signal = "🟢 BUY"
                        else:
                            prediction = "📉 BEARISH" 
                            signal = "🔴 SELL"
                        
                        print(f"{symbol:6} | ${current_price:8.2f} | {change:+6.2f}% | {prediction:10} | {signal}")
                        
                except Exception as e:
                    print(f"{symbol:6} | ERROR: {e}")
        
        print("=" * 80)
        print("⏳ Next update in 30 seconds... (Ctrl+C to stop)")
        time.sleep(30)

def main():
    print("Initializing AI Stock Predictor with 27 stocks...")
    time.sleep(2)
    quick_stock_analysis()

if __name__ == "__main__":
    main()