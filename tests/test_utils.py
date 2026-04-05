import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.helpers import *
from utils.validators import *
from utils.calculators import *
from utils.formatters import *


class TestHelpers:
    """Test cases for helper functions"""
    
    def test_setup_logging(self):
        """Test logging setup"""
        logger = setup_logging('test_logger', 'INFO')
        
        assert logger is not None
        assert logger.name == 'test_logger'
        assert logger.level == 20  # INFO level
        
    def test_timer_decorator(self):
        """Test timer decorator"""
        @timer
        def slow_function():
            import time
            time.sleep(0.1)
            return "done"
            
        result = slow_function()
        assert result == "done"
        
    def test_retry_decorator_success(self):
        """Test retry decorator with success"""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            return "success"
            
        result = flaky_function()
        assert result == "success"
        assert call_count == 1
        
    def test_retry_decorator_failure(self):
        """Test retry decorator with failure"""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.1)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
            
        with pytest.raises(ValueError):
            always_failing_function()
            
        assert call_count == 3  # Should retry 3 times
        
    def test_cache_result(self):
        """Test result caching"""
        call_count = 0
        
        @cache_result(ttl=1)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
            
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call - should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
        
    def test_is_market_open(self):
        """Test market open check"""
        # This test might fail depending on when it's run
        # We're mainly testing that the function runs without error
        result = is_market_open()
        assert isinstance(result, bool)
        
    def test_clean_dataframe(self):
        """Test DataFrame cleaning"""
        dirty_df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, np.nan, 8],
            'C': [9, 10, 11, 12]
        })
        
        clean_df = clean_dataframe(dirty_df)
        
        assert not clean_df.isnull().any().any()
        assert len(clean_df) > 0


class TestValidators:
    """Test cases for validation functions"""
    
    def test_validate_symbol(self):
        """Test symbol validation"""
        # Valid symbol
        is_valid, message = validate_symbol('AAPL')
        assert is_valid == True
        assert message == "Valid symbol"
        
        # Invalid symbol format
        is_valid, message = validate_symbol('invalid_symbol')
        assert is_valid == False
        assert "Symbol must be" in message
        
        # Empty symbol
        is_valid, message = validate_symbol('')
        assert is_valid == False
        
    def test_validate_date_range(self):
        """Test date range validation"""
        # Valid range
        is_valid, message = validate_date_range('2023-01-01', '2023-01-31')
        assert is_valid == True
        
        # Start after end
        is_valid, message = validate_date_range('2023-02-01', '2023-01-01')
        assert is_valid == False
        assert "Start date cannot be after end date" in message
        
        # Invalid format
        is_valid, message = validate_date_range('invalid', '2023-01-01')
        assert is_valid == False
        assert "Invalid date format" in message
        
    def test_validate_price(self):
        """Test price validation"""
        # Valid price
        is_valid, message = validate_price(150.50)
        assert is_valid == True
        
        # Negative price
        is_valid, message = validate_price(-10)
        assert is_valid == False
        
        # Too high price
        is_valid, message = validate_price(10000000)
        assert is_valid == False
        
    def test_validate_quantity(self):
        """Test quantity validation"""
        # Valid quantity
        is_valid, message = validate_quantity(100)
        assert is_valid == True
        
        # Fractional quantity
        is_valid, message = validate_quantity(100.5)
        assert is_valid == False
        
        # Negative quantity
        is_valid, message = validate_quantity(-50)
        assert is_valid == False


class TestCalculators:
    """Test cases for calculator functions"""
    
    def test_calculate_returns(self):
        """Test return calculation"""
        prices = pd.Series([100, 105, 103, 108])
        returns = calculate_returns(prices)
        
        assert isinstance(returns, pd.Series)
        assert len(returns) == 3  # One less than prices
        assert returns.iloc[0] == 0.05  # (105-100)/100
        
    def test_calculate_volatility(self):
        """Test volatility calculation"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        volatility = calculate_volatility(returns, annualize=False)
        
        assert volatility > 0
        assert isinstance(volatility, float)
        
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
        
    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation"""
        prices = pd.Series([100, 110, 90, 95, 120, 85])
        max_dd = calculate_max_drawdown(prices)
        
        assert max_dd < 0  # Should be negative
        assert isinstance(max_dd, float)
        
    def test_calculate_trade_pnl(self):
        """Test trade P&L calculation"""
        pnl = calculate_trade_pnl(
            entry_price=100, 
            exit_price=110, 
            quantity=10,
            commission=0.001
        )
        
        assert pnl['gross_pnl'] == 100  # (110-100)*10
        assert pnl['net_pnl'] < 100  # After commission
        assert 'pnl_percent' in pnl
        
    def test_calculate_position_size(self):
        """Test position size calculation"""
        size = calculate_position_size(
            portfolio_value=10000,
            entry_price=100,
            stop_loss_price=95,
            risk_per_trade=0.02
        )
        
        # Risk amount = 10000 * 0.02 = 200
        # Price risk = 100 - 95 = 5
        # Position size = 200 / 5 = 40
        assert size == 40


class TestFormatters:
    """Test cases for formatting functions"""
    
    def test_format_currency(self):
        """Test currency formatting"""
        # Regular amount
        assert format_currency(150.50) == "$150.50"
        
        # Large amount
        assert "M" in format_currency(1500000)
        
        # Negative amount
        assert format_currency(-50.25) == "$-50.25"
        
    def test_format_percentage(self):
        """Test percentage formatting"""
        assert format_percentage(0.05) == "5.00%"
        assert format_percentage(-0.03) == "-3.00%"
        assert format_percentage(0.12345, decimals=1) == "12.3%"
        
    def test_format_large_number(self):
        """Test large number formatting"""
        assert format_large_number(1500) == "1.50K"
        assert format_large_number(2500000) == "2.50M"
        assert format_large_number(3800000000) == "3.80B"
        
    def test_format_timestamp(self):
        """Test timestamp formatting"""
        timestamp = datetime(2023, 1, 15, 10, 30, 0)
        formatted = format_timestamp(timestamp)
        
        assert "2023-01-15" in formatted
        assert "10:30:00" in formatted
        
    def test_format_trade_signal(self):
        """Test trade signal formatting"""
        assert "🟢" in format_trade_signal('BUY')
        assert "🔴" in format_trade_signal('SELL')
        assert "🟡" in format_trade_signal('HOLD')
        
   def test_format_confidence(self):
        """Test confidence formatting"""
        assert "🟢" in format_confidence(0.9)
        assert "🟡" in format_confidence(0.7)
        assert "🔴" in format_confidence(0.5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])