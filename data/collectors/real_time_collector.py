import yfinance as yf
import pandas as pd
import asyncio
import websockets
import json
from datetime import datetime, timedelta
import time
import requests
from config import Config

class RealTimeDataCollector:
    def __init__(self):
        self.symbols = Config.DEFAULT_SYMBOLS
        self.data_buffer = {}
        self.historical_data = {}
        
    def get_historical_data(self, symbol, period="1y", interval="1d"):
        """Fetch historical data for training"""
        try:
            print(f"📊 Fetching historical data for {symbol}...")
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                print(f"❌ No data found for {symbol}")
                return None
                
            # Add symbol column
            data['Symbol'] = symbol
            data['Date'] = data.index
            
            print(f"✅ Successfully fetched {len(data)} records for {symbol}")
            self.historical_data[symbol] = data
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def get_multiple_historical_data(self, symbols=None, period="1y", interval="1d"):
        """Fetch historical data for multiple symbols"""
        if symbols is None:
            symbols = self.symbols
            
        all_data = {}
        for symbol in symbols:
            data = self.get_historical_data(symbol, period, interval)
            if data is not None:
                all_data[symbol] = data
                
        return all_data
    
    def get_intraday_data(self, symbol, interval="5m", days=7):
        """Fetch intraday data for real-time analysis"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=f"{days}d", interval=interval)
            data['Symbol'] = symbol
            return data
        except Exception as e:
            print(f"Error fetching intraday data for {symbol}: {e}")
            return None
    
    async def stream_real_time_data(self, symbol):
        """WebSocket streaming for real-time data (simulated)"""
        print(f"🔴 Starting real-time stream for {symbol}...")
        
        # Simulate real-time data (in production, use actual WebSocket)
        while True:
            try:
                # Get current price (simulated)
                current_price = await self.get_current_price(symbol)
                
                timestamp = datetime.now()
                
                if symbol not in self.data_buffer:
                    self.data_buffer[symbol] = []
                
                tick_data = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'price': current_price,
                    'volume': 1000,  # Simulated volume
                    'source': 'simulated'
                }
                
                self.data_buffer[symbol].append(tick_data)
                
                # Keep only last 1000 ticks
                if len(self.data_buffer[symbol]) > 1000:
                    self.data_buffer[symbol] = self.data_buffer[symbol][-1000:]
                
                # Wait before next update
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error in real-time stream for {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def get_current_price(self, symbol):
        """Get current stock price (simulated with random walk)"""
        import random
        base_price = 150  # Base price for simulation
        
        if symbol not in self.data_buffer or not self.data_buffer[symbol]:
            return base_price + random.uniform(-10, 10)
        
        last_price = self.data_buffer[symbol][-1]['price']
        change = random.uniform(-2, 2)
        return max(1, last_price + change)
    
    def get_latest_data(self, symbol, count=10):
        """Get latest data points for a symbol"""
        if symbol in self.data_buffer:
            return self.data_buffer[symbol][-count:]
        return []
    
    def save_data_to_csv(self, symbol, filename=None):
        """Save historical data to CSV"""
        if symbol in self.historical_data:
            if filename is None:
                filename = f"{symbol}_historical_data.csv"
            
            self.historical_data[symbol].to_csv(filename, index=False)
            print(f"💾 Saved {symbol} data to {filename}")
            return True
        return False

# Test function
def test_data_collector():
    """Test the data collector"""
    collector = RealTimeDataCollector()
    
    # Test historical data
    print("Testing historical data collection...")
    data = collector.get_historical_data('AAPL', period="6mo", interval="1d")
    
    if data is not None:
        print(f"Sample data:\n{data.head()}")
        print(f"Data columns: {data.columns.tolist()}")
        print(f"Data shape: {data.shape}")
    
    return collector

if __name__ == "__main__":
    test_data_collector()