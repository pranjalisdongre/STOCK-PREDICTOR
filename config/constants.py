"""
Application Constants
====================

Centralized constants used throughout the application.
"""

# Trading Constants
TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_DAY = 6.5  # 9:30 AM - 4:00 PM

# Signal Constants
SIGNAL_BUY = 'BUY'
SIGNAL_SELL = 'SELL'
SIGNAL_HOLD = 'HOLD'
SIGNAL_WEAK_BUY = 'WEAK_BUY'
SIGNAL_WEAK_SELL = 'WEAK_SELL'

SIGNAL_STRENGTH = {
    SIGNAL_BUY: 2,
    SIGNAL_SELL: -2,
    SIGNAL_WEAK_BUY: 1,
    SIGNAL_WEAK_SELL: -1,
    SIGNAL_HOLD: 0
}

# Market Hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Technical Indicator Constants
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_NEUTRAL_LOW = 40
RSI_NEUTRAL_HIGH = 60

# ML Model Constants
MODEL_TYPES = {
    'ensemble': 'EnsembleStockPredictor',
    'lstm': 'LSTMPredictor',
    'random_forest': 'RandomForestPredictor',
    'gradient_boosting': 'GradientBoostingPredictor'
}

# Feature Constants
PRICE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
TECHNICAL_FEATURES = [
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14', 'MACD',
    'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower',
    'ATR_14', 'OBV', 'Stoch_K', 'Stoch_D'
]

# Risk Levels
RISK_LEVEL_LOW = 'LOW'
RISK_LEVEL_MEDIUM = 'MEDIUM'
RISK_LEVEL_HIGH = 'HIGH'

RISK_THRESHOLDS = {
    RISK_LEVEL_LOW: 0.02,    # 2% volatility
    RISK_LEVEL_MEDIUM: 0.05, # 5% volatility
    RISK_LEVEL_HIGH: 0.08    # 8% volatility
}

# Portfolio Optimization Methods
OPTIMIZATION_SHARPE = 'sharpe'
OPTIMIZATION_MIN_VARIANCE = 'min_variance'
OPTIMIZATION_BLACK_LITTERMAN = 'black_litterman'

# Time Intervals
INTERVAL_1MIN = '1min'
INTERVAL_5MIN = '5min'
INTERVAL_15MIN = '15min'
INTERVAL_1HOUR = '1hour'
INTERVAL_1DAY = '1day'
INTERVAL_1WEEK = '1week'

INTERVAL_SECONDS = {
    INTERVAL_1MIN: 60,
    INTERVAL_5MIN: 300,
    INTERVAL_15MIN: 900,
    INTERVAL_1HOUR: 3600,
    INTERVAL_1DAY: 86400,
    INTERVAL_1WEEK: 604800
}

# Data Sources
DATA_SOURCE_YFINANCE = 'yfinance'
DATA_SOURCE_ALPHA_VANTAGE = 'alpha_vantage'
DATA_SOURCE_FINNHUB = 'finnhub'
DATA_SOURCE_POLYGON = 'polygon'

# Sentiment Constants
SENTIMENT_POSITIVE = 'POSITIVE'
SENTIMENT_NEGATIVE = 'NEGATIVE'
SENTIMENT_NEUTRAL = 'NEUTRAL'

SENTIMENT_THRESHOLDS = {
    SENTIMENT_POSITIVE: 0.1,
    SENTIMENT_NEGATIVE: -0.1
}

# Error Codes
ERROR_INSUFFICIENT_DATA = 'INSUFFICIENT_DATA'
ERROR_API_LIMIT_EXCEEDED = 'API_LIMIT_EXCEEDED'
ERROR_MODEL_NOT_TRAINED = 'MODEL_NOT_TRAINED'
ERROR_INVALID_SYMBOL = 'INVALID_SYMBOL'
ERROR_MARKET_CLOSED = 'MARKET_CLOSED'

# Performance Metrics
METRIC_MAE = 'mean_absolute_error'
METRIC_MSE = 'mean_squared_error'
METRIC_RMSE = 'root_mean_squared_error'
METRIC_R2 = 'r_squared'
METRIC_SHARPE = 'sharpe_ratio'
METRIC_MAX_DRAWDOWN = 'max_drawdown'
METRIC_WIN_RATE = 'win_rate'

# File Paths
MODELS_DIR = 'trained_models'
DATA_DIR = 'data'
LOGS_DIR = 'logs'
REPORTS_DIR = 'reports'

# Default Values
DEFAULT_PREDICTION_DAYS = 30
DEFAULT_BACKTEST_DAYS = 365
DEFAULT_PORTFOLIO_SIZE = 10
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# API Rate Limits (requests per minute)
RATE_LIMITS = {
    DATA_SOURCE_ALPHA_VANTAGE: 5,    # Free tier
    DATA_SOURCE_FINNHUB: 60,         # Free tier
    DATA_SOURCE_POLYGON: 5,          # Free tier
    DATA_SOURCE_YFINANCE: 2000       # No official limit
}