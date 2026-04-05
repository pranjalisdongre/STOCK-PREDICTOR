import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from trading.strategies.ml_enhanced_strategy import MLEnhancedTradingStrategy
from trading.strategies.risk_manager import RiskManager
from trading.backtesting.engine import BacktestingEngine
from trading.portfolio.optimizer import PortfolioOptimizer


class TestMLEnhancedTradingStrategy:
    """Test cases for MLEnhancedTradingStrategy"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.strategy = MLEnhancedTradingStrategy(initial_capital=10000, risk_per_trade=0.02)
        
    def test_initialization(self):
        """Test strategy initialization"""
        assert self.strategy.initial_capital == 10000
        assert self.strategy.risk_per_trade == 0.02
        assert self.strategy.portfolio_value == 10000
        assert self.strategy.cash == 10000
        assert isinstance(self.strategy.positions, dict)
        assert isinstance(self.strategy.trade_history, list)
        
    def test_calculate_position_size(self):
        """Test position size calculation"""
        position_size = self.strategy.calculate_position_size(
            current_price=100, stop_loss_price=95
        )
        
        assert position_size > 0
        # Risk amount = 10000 * 0.02 = 200
        # Price risk = 100 - 95 = 5
        # Position size = 200 / 5 = 40
        expected_size = (10000 * 0.02) / 5
        assert position_size == expected_size
        
    def test_calculate_position_size_insufficient_cash(self):
        """Test position size with insufficient cash"""
        # Set cash to very low amount
        self.strategy.cash = 10
        
        position_size = self.strategy.calculate_position_size(
            current_price=100, stop_loss_price=95
        )
        
        assert position_size == 0
        
    def test_generate_signals(self):
        """Test trading signal generation"""
        sample_data = pd.DataFrame({
            'Close': [100, 102, 101, 105, 103]
        })
        
        sample_predictions = {
            'ensemble_prediction': 110,
            'confidence': 0.8
        }
        
        sample_indicators = {
            'RSI_14': 65,
            'MACD': 0.5,
            'MACD_Signal': 0.3,
            'Price_vs_SMA20': 0.02,
            'BB_Position': 0.6
        }
        
        signal = self.strategy.generate_signals(
            sample_data, sample_predictions, sample_indicators
        )
        
        assert signal in ['BUY', 'SELL', 'HOLD', 'WEAK_BUY', 'WEAK_SELL']
        
    def test_get_ml_signal_high_confidence(self):
        """Test ML signal with high confidence"""
        predictions = {
            'ensemble_prediction': 110,  # 10% above current
            'confidence': 0.8
        }
        
        signal = self.strategy._get_ml_signal(predictions, 100)
        
        assert signal == 'BUY'
        
    def test_get_ml_signal_low_confidence(self):
        """Test ML signal with low confidence"""
        predictions = {
            'ensemble_prediction': 110,
            'confidence': 0.5  # Below threshold
        }
        
        signal = self.strategy._get_ml_signal(predictions, 100)
        
        assert signal == 'HOLD'
        
    def test_execute_trade_buy(self):
        """Test buy trade execution"""
        sample_data = pd.DataFrame({
            'Close': [100, 102, 101, 105, 103]
        })
        
        sample_predictions = {
            'ensemble_prediction': 110,
            'confidence': 0.8
        }
        
        # Mock signal generation to return BUY
        with patch.object(self.strategy, 'generate_signals', return_value='BUY'):
            trade = self.strategy.execute_trade(
                'AAPL', 'BUY', sample_data, sample_predictions
            )
            
            if trade:  # Trade might be None if insufficient cash
                assert trade['action'] == 'BUY'
                assert trade['symbol'] == 'AAPL'
                assert 'AAPL' in self.strategy.positions
                
    def test_execute_trade_sell(self):
        """Test sell trade execution"""
        # First create a position
        self.strategy.positions['AAPL'] = {
            'entry_price': 100,
            'position_size': 10,
            'entry_time': datetime.now()
        }
        self.strategy.cash = 5000  # Reduce cash to simulate having position
        
        sample_data = pd.DataFrame({
            'Close': [100, 102, 101, 105, 110]  # Price increased
        })
        
        sample_predictions = {
            'ensemble_prediction': 95,  # Predict price drop
            'confidence': 0.8
        }
        
        # Mock signal generation to return SELL
        with patch.object(self.strategy, 'generate_signals', return_value='SELL'):
            trade = self.strategy.execute_trade(
                'AAPL', 'SELL', sample_data, sample_predictions
            )
            
            if trade:
                assert trade['action'] == 'SELL'
                assert trade['symbol'] == 'AAPL'
                assert 'pnl' in trade
                
    def test_get_portfolio_summary(self):
        """Test portfolio summary generation"""
        # Add a sample position
        self.strategy.positions['AAPL'] = {
            'entry_price': 100,
            'position_size': 10,
            'entry_time': datetime.now()
        }
        self.strategy.cash = 5000
        
        summary = self.strategy.get_portfolio_summary()
        
        assert 'portfolio_value' in summary
        assert 'cash' in summary
        assert 'open_positions' in summary
        assert 'total_trades' in summary
        assert summary['open_positions'] == 1
        
    def test_get_performance_metrics(self):
        """Test performance metrics calculation"""
        # Add some sample trades
        self.strategy.trade_history = [
            {
                'symbol': 'AAPL', 'action': 'SELL', 'price': 110, 'size': 10,
                'timestamp': datetime.now(), 'pnl': 100, 'pnl_percent': 0.1
            },
            {
                'symbol': 'GOOGL', 'action': 'SELL', 'price': 2500, 'size': 2,
                'timestamp': datetime.now(), 'pnl': -50, 'pnl_percent': -0.01
            }
        ]
        
        metrics = self.strategy.get_performance_metrics()
        
        assert 'total_pnl' in metrics
        assert 'win_rate' in metrics
        assert 'sharpe_ratio' in metrics
        assert metrics['total_pnl'] == 50  # 100 - 50


class TestRiskManager:
    """Test cases for RiskManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.risk_manager = RiskManager()
        
    def test_initialization(self):
        """Test risk manager initialization"""
        assert self.risk_manager.max_portfolio_risk == 0.10
        assert self.risk_manager.max_position_risk == 0.02
        assert self.risk_manager.max_drawdown == 0.15
        
    def test_assess_market_conditions(self):
        """Test market condition assessment"""
        sample_market_data = pd.DataFrame({
            'Close': np.random.normal(100, 5, 50).cumsum() + 1000
        })
        
        regime = self.risk_manager.assess_market_conditions(sample_market_data)
        
        assert regime in ['HIGH_VOLATILITY', 'LOW_VOLATILITY', 'BULLISH', 'BEARISH', 'NORMAL']
        
    def test_calculate_position_limits(self):
        """Test position limit calculation"""
        limits = self.risk_manager.calculate_position_limits('HIGH_VOLATILITY', 0.03)
        
        assert 'position_limit' in limits
        assert 'leverage_multiplier' in limits
        assert 'stop_loss_adjustment' in limits
        assert limits['position_limit'] < 0.02  # Should be reduced in high volatility
        
    def test_validate_trade_approved(self):
        """Test trade validation (approved)"""
        trade_signal = {
            'symbol': 'AAPL',
            'position_size': 10,
            'price': 150
        }
        
        portfolio = {
            'portfolio_value': 10000,
            'positions': {},
            'peak_value': 11000
        }
        
        market_data = pd.DataFrame({'Close': [150, 151, 152]})
        symbol_data = pd.DataFrame({'Close': [150, 151, 152]})
        
        validation = self.risk_manager.validate_trade(
            trade_signal, portfolio, market_data, symbol_data
        )
        
        assert validation['approved'] == True
        assert 'adjusted_size' in validation
        
    def test_validate_trade_rejected(self):
        """Test trade validation (rejected)"""
        trade_signal = {
            'symbol': 'AAPL',
            'position_size': 1000,  # Very large position
            'price': 150
        }
        
        portfolio = {
            'portfolio_value': 10000,
            'positions': {'AAPL': {'position_size': 50, 'entry_price': 140}},
            'peak_value': 11000
        }
        
        market_data = pd.DataFrame({'Close': [150, 151, 152]})
        symbol_data = pd.DataFrame({'Close': [150, 151, 152]})
        
        validation = self.risk_manager.validate_trade(
            trade_signal, portfolio, market_data, symbol_data
        )
        
        # Should be rejected due to over-concentration
        assert validation['approved'] == False
        assert len(validation['reasons']) > 0
        
    def test_get_risk_report(self):
        """Test risk report generation"""
        portfolio = {
            'portfolio_value': 10000,
            'positions': {'AAPL': {'position_size': 10, 'entry_price': 150}},
            'peak_value': 11000
        }
        
        market_data = pd.DataFrame({
            'Close': np.random.normal(100, 5, 100).cumsum() + 1000
        })
        
        report = self.risk_manager.get_risk_report(portfolio, market_data)
        
        assert 'portfolio_risk' in report
        assert 'market_conditions' in report
        assert 'risk_metrics' in report
        assert 'recommendations' in report


class TestBacktestingEngine:
    """Test cases for BacktestingEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.backtester = BacktestingEngine(
            initial_capital=10000, commission=0.001, slippage=0.001
        )
        
    def test_initialization(self):
        """Test backtesting engine initialization"""
        assert self.backtester.initial_capital == 10000
        assert self.backtester.commission == 0.001
        assert self.backtester.slippage == 0.001
        
    def test_run_backtest(self):
        """Test backtest execution"""
        # Create sample price data
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105],
            'High': [105, 106, 107, 108, 109, 110],
            'Low': [98, 99, 100, 101, 102, 103],
            'Close': [103, 104, 105, 106, 107, 108],
            'Volume': [1000000, 2000000, 1500000, 1800000, 1200000, 1300000]
        })
        
        from trading.strategies.ml_enhanced_strategy import MLEnhancedTradingStrategy
        
        performance = self.backtester.run_backtest(
            sample_data, MLEnhancedTradingStrategy, 'TEST'
        )
        
        assert isinstance(performance, dict)
        assert 'total_return' in performance
        assert 'sharpe_ratio' in performance
        assert 'max_drawdown' in performance
        
    def test_run_comparative_analysis(self):
        """Test comparative backtest analysis"""
        sample_data_dict = {
            'AAPL': pd.DataFrame({
                'Open': [100, 101, 102, 103],
                'High': [105, 106, 107, 108],
                'Low': [98, 99, 100, 101],
                'Close': [103, 104, 105, 106],
                'Volume': [1000000, 2000000, 1500000, 1800000]
            }),
            'GOOGL': pd.DataFrame({
                'Open': [2500, 2510, 2520, 2530],
                'High': [2550, 2560, 2570, 2580],
                'Low': [2480, 2490, 2500, 2510],
                'Close': [2530, 2540, 2550, 2560],
                'Volume': [500000, 600000, 550000, 580000]
            })
        }
        
        from trading.strategies.ml_enhanced_strategy import MLEnhancedTradingStrategy
        
        results, report = self.backtester.run_comparative_analysis(
            sample_data_dict, MLEnhancedTradingStrategy
        )
        
        assert isinstance(results, dict)
        assert isinstance(report, dict)
        assert 'AAPL' in results
        assert 'GOOGL' in results
        assert 'summary' in report


class TestPortfolioOptimizer:
    """Test cases for PortfolioOptimizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        
    def test_calculate_portfolio_metrics(self):
        """Test portfolio metrics calculation"""
        weights = [0.5, 0.3, 0.2]
        expected_returns = [0.1, 0.12, 0.15]
        cov_matrix = pd.DataFrame({
            'A': [0.04, 0.02, 0.01],
            'B': [0.02, 0.06, 0.02],
            'C': [0.01, 0.02, 0.03]
        }, index=['A', 'B', 'C'])
        
        metrics = self.optimizer.calculate_portfolio_metrics(
            weights, expected_returns, cov_matrix
        )
        
        assert 'return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert metrics['return'] > 0
        
    def test_optimize_sharpe_ratio(self):
        """Test Sharpe ratio optimization"""
        expected_returns = pd.Series([0.1, 0.12, 0.15], index=['A', 'B', 'C'])
        cov_matrix = pd.DataFrame({
            'A': [0.04, 0.02, 0.01],
            'B': [0.02, 0.06, 0.02],
            'C': [0.01, 0.02, 0.03]
        }, index=['A', 'B', 'C'])
        
        result = self.optimizer.optimize_sharpe_ratio(
            expected_returns, cov_matrix, max_allocation=0.5
        )
        
        assert result is not None
        assert 'weights' in result
        assert 'metrics' in result
        assert abs(sum(result['weights'].values()) - 1.0) < 0.01  # Sum to ~1
        
    def test_optimize_minimum_variance(self):
        """Test minimum variance optimization"""
        cov_matrix = pd.DataFrame({
            'A': [0.04, 0.02, 0.01],
            'B': [0.02, 0.06, 0.02],
            'C': [0.01, 0.02, 0.03]
        }, index=['A', 'B', 'C'])
        
        result = self.optimizer.optimize_minimum_variance(cov_matrix, max_allocation=0.5)
        
        assert result is not None
        assert 'weights' in result
        assert 'metrics' in result
        assert result['metrics']['volatility'] > 0
        
    def test_rebalance_portfolio(self):
        """Test portfolio rebalancing"""
        current_weights = {'AAPL': 0.4, 'GOOGL': 0.6}
        target_weights = {'AAPL': 0.5, 'GOOGL': 0.5}
        prices = {'AAPL': 150, 'GOOGL': 2500}
        
        rebalance_result = self.optimizer.rebalance_portfolio(
            current_weights, target_weights, prices, transaction_cost=0.001
        )
        
        assert 'trades' in rebalance_result
        assert 'total_transaction_cost' in rebalance_result
        assert 'rebalancing_cost_percent' in rebalance_result
        assert 'AAPL' in rebalance_result['trades']
        assert 'GOOGL' in rebalance_result['trades']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])