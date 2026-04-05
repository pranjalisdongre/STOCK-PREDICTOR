# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'your_news_api_key')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', 'your_finnhub_key')
    
    # Stock Symbols to track
    DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NFLX']
    
    # Data Collection
    UPDATE_INTERVAL = 60  # seconds
    HISTORICAL_DAYS = 365
    
    # ML Settings
    SEQUENCE_LENGTH = 60
    TRAIN_TEST_SPLIT = 0.8
    
    # Trading
    INITIAL_CAPITAL = 10000
    RISK_PER_TRADE = 0.02  # 2%
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///stock_data.db')