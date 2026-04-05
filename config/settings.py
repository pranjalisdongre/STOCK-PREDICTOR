import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class with default settings"""
    
    # Project Metadata
    PROJECT_NAME = "AI Stock Predictor"
    PROJECT_VERSION = "1.0.0"
    PROJECT_DESCRIPTION = "Intelligent stock trading platform powered by machine learning"
    
    # API Keys (Get free keys from these services)
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'your_news_api_key_here')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', 'your_finnhub_key_here')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', 'your_polygon_key_here')
    
    # Stock Symbols Configuration
    DEFAULT_SYMBOLS = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',        # Tech
        'META', 'NFLX', 'NVDA', 'AMD', 'INTC',          # More Tech
        'JPM', 'BAC', 'GS', 'MS',                       # Financials
        'JNJ', 'PFE', 'UNH', 'MRK',                     # Healthcare
        'XOM', 'CVX', 'COP',                            # Energy
        'WMT', 'TGT', 'COST',                           # Retail
        'SPY', 'QQQ', 'DIA'                             # ETFs
    ]
    
    # Data Collection Settings
    DATA_UPDATE_INTERVAL = 60  # seconds
    HISTORICAL_DAYS = 365
    REAL_TIME_UPDATE_INTERVAL = 5  # seconds for real-time data
    
    # ML Model Configuration
    SEQUENCE_LENGTH = 60
    TRAIN_TEST_SPLIT = 0.8
    PREDICTION_LOOKAHEAD = 1  # Predict next day
    MODEL_RETRAIN_INTERVAL = timedelta(days=7)  # Retrain models weekly
    
    # Trading Configuration
    INITIAL_CAPITAL = 10000.0
    RISK_PER_TRADE = 0.02  # 2% risk per trade
    MAX_PORTFOLIO_RISK = 0.10  # 10% max portfolio risk
    COMMISSION_RATE = 0.001  # 0.1% commission
    SLIPPAGE = 0.001  # 0.1% slippage
    
    # Risk Management
    STOP_LOSS_PERCENT = 0.04  # 4% stop loss
    TAKE_PROFIT_PERCENT = 0.08  # 8% take profit
    MAX_POSITION_SIZE = 0.3  # Max 30% in single position
    MAX_DRAWDOWN = 0.15  # 15% max drawdown
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///stock_predictor.db')
    
    # Flask Web Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # Caching Configuration
    CACHE_TYPE = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'stock_predictor.log'
    
    # Performance Settings
    MAX_WORKERS = 4
    BATCH_SIZE = 100
    REQUEST_TIMEOUT = 30  # seconds
    
    # Feature Flags
    ENABLE_REAL_TIME_TRADING = False  # Set to True for live trading (CAUTION!)
    ENABLE_AUTO_RETRAINING = True
    ENABLE_EMAIL_ALERTS = False
    
    # Backtesting
    BACKTEST_INITIAL_CAPITAL = 10000
    BACKTEST_COMMISSION = 0.001
    BACKTEST_SLIPPAGE = 0.001
    
    # Portfolio Optimization
    RISK_FREE_RATE = 0.02  # 2% risk-free rate
    OPTIMIZATION_METHOD = 'sharpe'  # 'sharpe', 'min_variance', 'black_litterman'
    MAX_ALLOCATION_PER_ASSET = 0.3  # 30% max allocation
    
    # Sentiment Analysis
    SENTIMENT_ANALYSIS_ENABLED = True
    NEWS_LOOKBACK_DAYS = 7
    MIN_SENTIMENT_CONFIDENCE = 0.6
    
    # Technical Indicators
    ENABLED_INDICATORS = [
        'SMA', 'EMA', 'MACD', 'RSI', 'BBANDS', 'ATR', 
        'STOCH', 'ADX', 'OBV', 'CCI', 'WILLR'
    ]
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        # Check API keys
        if cls.ALPHA_VANTAGE_API_KEY == 'demo':
            errors.append("Using Alpha Vantage demo key - limited functionality")
        
        if cls.NEWS_API_KEY == 'your_news_api_key_here':
            errors.append("News API key not configured - sentiment analysis disabled")
            cls.SENTIMENT_ANALYSIS_ENABLED = False
        
        # Validate risk parameters
        if cls.RISK_PER_TRADE > cls.MAX_PORTFOLIO_RISK:
            errors.append("Risk per trade cannot exceed max portfolio risk")
        
        if cls.MAX_POSITION_SIZE > 0.5:
            errors.append("Max position size too high - consider lowering to 0.3")
        
        return errors
    
    @classmethod
    def get_symbol_categories(cls):
        """Get symbols categorized by sector"""
        return {
            'technology': ['AAPL', 'GOOGL', 'MSFT', 'META', 'NFLX', 'NVDA', 'AMD', 'INTC'],
            'financials': ['JPM', 'BAC', 'GS', 'MS'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'MRK'],
            'energy': ['XOM', 'CVX', 'COP'],
            'retail': ['AMZN', 'WMT', 'TGT', 'COST'],
            'automotive': ['TSLA'],
            'etfs': ['SPY', 'QQQ', 'DIA']
        }


class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    ENABLE_REAL_TIME_TRADING = False  # Always false in development
    
    # Faster updates for development
    DATA_UPDATE_INTERVAL = 30
    REAL_TIME_UPDATE_INTERVAL = 10
    
    # Smaller dataset for faster development
    HISTORICAL_DAYS = 90
    SEQUENCE_LENGTH = 30


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Ensure secure secret key in production
    def __init__(self):
        super().__init__()
        self.SECRET_KEY = os.getenv('SECRET_KEY')
        if not self.SECRET_KEY or self.SECRET_KEY == 'dev-secret-key-change-in-production':
            raise ValueError("SECRET_KEY must be set in production environment")
    
    # More conservative settings for production
    RISK_PER_TRADE = 0.015  # 1.5% risk per trade
    MAX_PORTFOLIO_RISK = 0.08  # 8% max portfolio risk


class TestingConfig(Config):
    """Testing environment configuration"""
    TESTING = True
    DEBUG = True
    DATABASE_URL = 'sqlite:///:memory:'
    
    # Fast execution for tests
    DATA_UPDATE_INTERVAL = 1
    HISTORICAL_DAYS = 30
    SEQUENCE_LENGTH = 10


def get_config():
    """Get appropriate configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    return configs.get(env, DevelopmentConfig)


# Global config instance
current_config = get_config()