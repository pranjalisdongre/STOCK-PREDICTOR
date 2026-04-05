import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from data.collectors.real_time_collector import RealTimeDataCollector
from data.collectors.news_collector import NewsSentimentAnalyzer
from data.processors.technical_indicators import TechnicalIndicatorProcessor
from data.processors.feature_engineer import FeatureEngineer


class TestRealTimeDataCollector:
    """Test cases for RealTimeDataCollector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.collector = RealTimeDataCollector()
        
    def test_initialization(self):
        """Test collector initialization"""
        assert self.collector is not None
        assert hasattr(self.collector, 'symbols')
        assert isinstance(self.collector.symbols, list)
        assert len(self.collector.symbols) > 0
        
    @patch('data.collectors.real_time_collector.yf.Ticker')
    def test_get_historical_data_success(self, mock_ticker):
        """Test successful historical data retrieval"""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [150, 151, 152],
            'High': [155, 156, 157], 
            'Low': [148, 149, 150],
            'Close': [153, 154, 155],
            'Volume': [1000000, 2000000, 1500000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_ticker.return_value.history.return_value = mock_data
        
        result = self.collector.get_historical_data('AAPL', period='1mo')
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'Symbol' in result.columns
        assert result['Symbol'].iloc[0] == 'AAPL'
        
    @patch('data.collectors.real_time_collector.yf.Ticker')
    def test_get_historical_data_failure(self, mock_ticker):
        """Test historical data retrieval failure"""
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        
        result = self.collector.get_historical_data('INVALID', period='1mo')
        
        assert result is None
        
    def test_get_multiple_historical_data(self):
        """Test multiple symbol data retrieval"""
        with patch.object(self.collector, 'get_historical_data') as mock_get:
            mock_get.return_value = pd.DataFrame({'Close': [100, 101, 102]})
            
            results = self.collector.get_multiple_historical_data(['AAPL', 'GOOGL'])
            
            assert isinstance(results, dict)
            assert 'AAPL' in results
            assert 'GOOGL' in results
            assert mock_get.call_count == 2


class TestNewsSentimentAnalyzer:
    """Test cases for NewsSentimentAnalyzer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = NewsSentimentAnalyzer()
        
    @patch('data.collectors.news_collector.requests.get')
    def test_fetch_news_success(self, mock_get):
        """Test successful news fetching"""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'articles': [
                {
                    'title': 'Apple Reports Strong Earnings',
                    'description': 'Apple exceeds expectations',
                    'source': {'name': 'Reuters'},
                    'publishedAt': '2023-01-01T10:00:00Z',
                    'url': 'https://example.com/1'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        result = self.analyzer.fetch_news('AAPL', days=1)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'symbol' in result.columns
        assert 'title' in result.columns
        
    @patch('data.collectors.news_collector.requests.get')
    def test_fetch_news_api_error(self, mock_get):
        """Test news fetching with API error"""
        mock_response = Mock()
        mock_response.status_code = 401  # Unauthorized
        mock_get.return_value = mock_response
        
        result = self.analyzer.fetch_news('AAPL')
        
        assert isinstance(result, pd.DataFrame)
        # Should return sample data on error
        
    def test_calculate_daily_sentiment(self):
        """Test daily sentiment calculation"""
        # Create sample news data
        sample_news = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'custom_sentiment': [0.5, -0.2],
            'published_at': [datetime.now(), datetime.now() - timedelta(hours=1)]
        })
        
        with patch.object(self.analyzer, 'fetch_news', return_value=sample_news):
            result = self.analyzer.calculate_daily_sentiment('AAPL')
            
            assert result is not None
            assert 'symbol' in result
            assert 'sentiment_score' in result
            assert 'article_count' in result


class TestTechnicalIndicatorProcessor:
    """Test cases for TechnicalIndicatorProcessor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = TechnicalIndicatorProcessor()
        
    def test_calculate_all_indicators(self):
        """Test technical indicator calculation"""
        # Create sample price data
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [98, 99, 100, 101, 102],
            'Close': [103, 104, 105, 106, 107],
            'Volume': [1000000, 2000000, 1500000, 1800000, 1200000]
        })
        
        result = self.processor.calculate_all_indicators(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'SMA_20' in result.columns
        assert 'RSI_14' in result.columns
        assert 'MACD' in result.columns
        
    def test_empty_dataframe(self):
        """Test indicator calculation with empty data"""
        empty_df = pd.DataFrame()
        
        result = self.processor.calculate_all_indicators(empty_df)
        
        assert result.empty
        
    def test_get_trading_signals(self):
        """Test trading signal generation"""
        sample_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'SMA_20': [99, 100, 101, 102, 103],
            'SMA_50': [98, 99, 100, 101, 102],
            'RSI_14': [45, 50, 55, 60, 65],
            'MACD': [0.1, 0.2, 0.3, 0.4, 0.5],
            'MACD_Signal': [0.05, 0.15, 0.25, 0.35, 0.45],
            'BB_Upper': [105, 106, 107, 108, 109],
            'BB_Lower': [95, 96, 97, 98, 99]
        })
        
        signals = self.processor.get_trading_signals(sample_data)
        
        assert isinstance(signals, dict)
        assert 'composite' in signals


class TestFeatureEngineer:
    """Test cases for FeatureEngineer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engineer = FeatureEngineer()
        
    def test_create_advanced_features(self):
        """Test advanced feature engineering"""
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [98, 99, 100, 101, 102],
            'Close': [103, 104, 105, 106, 107],
            'Volume': [1000000, 2000000, 1500000, 1800000, 1200000],
            'Date': pd.date_range('2023-01-01', periods=5)
        })
        
        result = self.engineer.create_advanced_features(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'price_change' in result.columns
        assert 'volume_change' in result.columns
        assert 'target_return_1' in result.columns
        
    def test_feature_selection(self):
        """Test feature selection functionality"""
        sample_data = pd.DataFrame({
            'Close': range(100),
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100),
            'Date': pd.date_range('2023-01-01', periods=100)
        })
        
        # Add engineered features
        sample_data = self.engineer.create_advanced_features(sample_data)
        
        selected_features = self.engineer.select_best_features(
            sample_data, 'Close', k=2
        )
        
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])