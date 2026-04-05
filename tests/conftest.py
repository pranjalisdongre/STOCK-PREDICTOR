"""
Pytest configuration and fixtures for AI Stock Predictor tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_price_data():
    """Fixture providing sample price data for tests."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'Open': np.random.normal(150, 5, 100).cumsum() + 1500,
        'High': np.random.normal(152, 5, 100).cumsum() + 1500,
        'Low': np.random.normal(148, 5, 100).cumsum() + 1500,
        'Close': np.random.normal(150, 5, 100).cumsum() + 1500,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)


@pytest.fixture
def sample_returns_data():
    """Fixture providing sample returns data for tests."""
    return pd.Series(np.random.normal(0.001, 0.02, 100))


@pytest.fixture
def sample_portfolio():
    """Fixture providing sample portfolio data for tests."""
    return {
        'portfolio_value': 10000,
        'cash': 5000,
        'positions': {
            'AAPL': {
                'entry_price': 150,
                'position_size': 10,
                'entry_time': datetime.now() - timedelta(days=5)
            },
            'GOOGL': {
                'entry_price': 2500,
                'position_size': 2,
                'entry_time': datetime.now() - timedelta(days=3)
            }
        },
        'peak_value': 11000
    }


@pytest.fixture
def sample_trade_signal():
    """Fixture providing sample trade signal for tests."""
    return {
        'symbol': 'AAPL',
        'signal': 'BUY',
        'position_size': 10,
        'price': 150,
        'confidence': 0.75
    }


@pytest.fixture
def sample_ml_predictions():
    """Fixture providing sample ML predictions for tests."""
    return {
        'ensemble_prediction': 155.25,
        'confidence': 0.82,
        'individual_predictions': {
            'random_forest': 154.80,
            'gradient_boosting': 155.50,
            'lstm': 155.45
        },
        'timestamp': datetime.now()
    }


@pytest.fixture
def sample_news_data():
    """Fixture providing sample news data for tests."""
    return pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'GOOGL'],
        'title': [
            'Apple Reports Strong Earnings',
            'New iPhone Launch Successful', 
            'Google AI Breakthrough'
        ],
        'published_at': [
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=1),
            datetime.now() - timedelta(hours=3)
        ],
        'custom_sentiment': [0.8, 0.6, 0.7],
        'vader_compound': [0.9, 0.5, 0.8]
    })


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set test environment variables
    os.environ['FLASK_ENV'] = 'testing'
    os.environ['ALPHA_VANTAGE_API_KEY'] = 'test_key'
    
    # Create test directories
    test_dirs = ['logs', 'data', 'trained_models', 'reports']
    for dir_name in test_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    yield
    
    # Cleanup after tests
    # Remove test files if needed


def pytest_configure(config):
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "web: mark test as web application test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Skip slow tests by default
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)