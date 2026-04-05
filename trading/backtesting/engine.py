import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BacktestingEngine:
    def __init__(self, initial_capital=10000, commission=0.001, slippage=0.001):
        self.initial_capital = initial_capital
        self.commission = commission  # 0.1% commission
        self.slippage = slippage      # 0.1% slippage
        self.results = {}
        self.trade_log = []
        
    def run_backtest(self, data, strategy, symbol, **strategy_params):
        """Run backtest for a single symbol"""
        print(f"🔍 Running backtest for {symbol}...")
        
        # Initialize strategy
        trading_strategy = strategy(initial_capital=self.initial_capital, **strategy_params)
        
        # Prepare data
        data = data.copy().sort_index()
        
        # Track portfolio values
        portfolio_values = []
        dates = []
        
        # Iterate through data
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            current_date = data.index[i]
            
            if len(current_data) < 2:  # Need at least 2 data points
                continue
            
            # Generate trading signal (simplified - would use actual ML predictions)
            signal = self._generate_simulated_signal(current_data, trading_strategy)
            
            # Execute trade
            if signal in ['BUY', 'SELL']:
                trade = self._execute_trade_simulation(
                    trading_strategy, symbol, signal, current_data.iloc[-1], current_date
                )
                if trade:
                    self.trade_log.append(trade)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(trading_strategy, current_data.iloc[-1], symbol)
            portfolio_values.append(portfolio_value)
            dates.append(current_date)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(
            portfolio_values, dates, trading_strategy, symbol
        )
        
        self.results[symbol] = performance
        return performance
    
    def _generate_simulated_signal(self, data, strategy):
        """Generate simulated trading signals for backtesting"""
        # This is a simplified version - in production, use actual ML predictions
        if len(data) < 20:
            return 'HOLD'
        
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        
        # Simple momentum strategy for simulation
        returns_5d = (current_price / data['Close'].iloc[-5] - 1) if len(data) >= 5 else 0
        returns_20d = (current_price / data['Close'].iloc[-20] - 1) if len(data) >= 20 else 0
        
        # Simulate ML predictions
        simulated_predictions = {
            'ensemble_prediction': current_price * (1 + returns_5d * 0.7),
            'confidence': min(abs(returns_5d) * 10, 0.9),
            'technical_indicators': {
                'RSI_14': min(max(30 + returns_5d * 1000, 20), 80),
                'MACD': returns_5d * 100,
                'Price_vs_SMA20': returns_20d
            }
        }
        
        # Generate signal using strategy
        signal = strategy.generate_signals(
            data, simulated_predictions, simulated_predictions['technical_indicators']
        )
        
        return signal
    
    def _execute_trade_simulation(self, strategy, symbol, signal, current_bar, current_date):
        """Execute trade simulation with commissions and slippage"""
        current_price = current_bar['Close']
        
        # Apply slippage
        if signal == 'BUY':
            execution_price = current_price * (1 + self.slippage)
        else:  # SELL
            execution_price = current_price * (1 - self.slippage)
        
        # Simulate trade execution
        if signal == 'BUY' and symbol not in strategy.positions:
            # Calculate position size
            stop_loss = execution_price * 0.96  # 4% stop loss
            position_size = strategy.calculate_position_size(execution_price, stop_loss)
            
            if position_size > 0:
                # Apply commission
                commission_cost = position_size * execution_price * self.commission
                total_cost = position_size * execution_price + commission_cost
                
                if total_cost <= strategy.cash:
                    # Execute buy
                    strategy.positions[symbol] = {
                        'entry_price': execution_price,
                        'position_size': position_size,
                        'entry_time': current_date,
                        'stop_loss': stop_loss,
                        'commission': commission_cost
                    }
                    strategy.cash -= total_cost
                    
                    return {
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': execution_price,
                        'size': position_size,
                        'timestamp': current_date,
                        'commission': commission_cost
                    }
        
        elif signal == 'SELL' and symbol in strategy.positions:
            position = strategy.positions[symbol]
            
            # Calculate P&L
            pnl = (execution_price - position['entry_price']) * position['position_size']
            pnl_percent = (execution_price - position['entry_price']) / position['entry_price']
            
            # Apply commission
            commission_cost = position['position_size'] * execution_price * self.commission
            proceeds = position['position_size'] * execution_price - commission_cost
            
            # Update portfolio
            strategy.cash += proceeds
            del strategy.positions[symbol]
            
            return {
                'symbol': symbol,
                'action': 'SELL',
                'price': execution_price,
                'size': position['position_size'],
                'timestamp': current_date,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'commission': commission_cost,
                'holding_period': (current_date - position['entry_time']).days
            }
        
        return None
    
    def _calculate_portfolio_value(self, strategy, current_bar, symbol):
        """Calculate current portfolio value"""
        cash = strategy.cash
        positions_value = 0
        
        for pos_symbol, position in strategy.positions.items():
            # Use current price for open positions
            if pos_symbol == symbol:
                positions_value += position['position_size'] * current_bar['Close']
            else:
                # For other symbols, use entry price (simplified)
                positions_value += position['position_size'] * position['entry_price']
        
        return cash + positions_value
    
    def _calculate_performance_metrics(self, portfolio_values, dates, strategy, symbol):
        """Calculate comprehensive performance metrics"""
        if len(portfolio_values) < 2:
            return {}
        
        # Basic metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate returns
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        running_max = pd.Series(portfolio_values).expanding().max()
        drawdown = (pd.Series(portfolio_values) - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        closed_trades = [t for t in self.trade_log if t.get('action') == 'SELL' and t['symbol'] == symbol]
        
        if closed_trades:
            winning_trades = [t for t in closed_trades if t['pnl'] > 0]
            losing_trades = [t for t in closed_trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(closed_trades)
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Strategy-specific metrics
        strategy_metrics = strategy.get_performance_metrics()
        
        return {
            'symbol': symbol,
            'total_return': total_return,
            'annualized_return': total_return * (252 / len(portfolio_values)) if len(portfolio_values) > 0 else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(closed_trades),
            'avg_trade_return': np.mean([t.get('pnl_percent', 0) for t in closed_trades]) if closed_trades else 0,
            'final_portfolio_value': final_value,
            'strategy_metrics': strategy_metrics,
            'calmar_ratio': -total_return / max_drawdown if max_drawdown < 0 else 0,
            'sortino_ratio': self._calculate_sortino_ratio(returns)
        }
    
    def _calculate_sortino_ratio(self, returns):
        """Calculate Sortino ratio (downside risk only)"""
        if len(returns) == 0:
            return 0
        
        downside_returns = returns[returns < 0]
        downside_risk = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        if downside_risk == 0:
            return 0
        
        return (returns.mean() * 252) / downside_risk
    
    def run_comparative_analysis(self, data_dict, strategy, benchmark_symbol='SPY', **strategy_params):
        """Run backtest across multiple symbols and compare with benchmark"""
        comparative_results = {}
        
        print("📊 Running comparative backtest analysis...")
        
        # Test strategy on each symbol
        for symbol, data in data_dict.items():
            if len(data) < 50:  # Need sufficient data
                continue
            
            result = self.run_backtest(data, strategy, symbol, **strategy_params)
            comparative_results[symbol] = result
        
        # Calculate benchmark performance
        if benchmark_symbol in data_dict:
            benchmark_data = data_dict[benchmark_symbol]
            benchmark_return = (benchmark_data['Close'].iloc[-1] / benchmark_data['Close'].iloc[0] - 1)
            
            comparative_results['Benchmark'] = {
                'symbol': benchmark_symbol,
                'total_return': benchmark_return,
                'volatility': benchmark_data['Close'].pct_change().std() * np.sqrt(252),
                'max_drawdown': self._calculate_benchmark_drawdown(benchmark_data)
            }
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report(comparative_results)
        
        return comparative_results, comparison_report
    
    def _calculate_benchmark_drawdown(self, benchmark_data):
        """Calculate maximum drawdown for benchmark"""
        prices = benchmark_data['Close']
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max
        return drawdown.min()
    
    def _generate_comparison_report(self, results):
        """Generate comparative performance report"""
        report = {
            'timestamp': datetime.now(),
            'summary': {},
            'ranking': {},
            'recommendations': []
        }
        
        # Calculate summary statistics
        returns = [result['total_return'] for result in results.values() if 'total_return' in result]
        sharpe_ratios = [result.get('sharpe_ratio', 0) for result in results.values()]
        
        if returns:
            report['summary'] = {
                'avg_return': np.mean(returns),
                'best_return': max(returns),
                'worst_return': min(returns),
                'avg_sharpe': np.mean(sharpe_ratios),
                'positive_returns': sum(1 for r in returns if r > 0)
            }
        
        # Rank strategies by Sharpe ratio
        ranked_results = sorted(
            [(symbol, result) for symbol, result in results.items() if 'sharpe_ratio' in result],
            key=lambda x: x[1]['sharpe_ratio'],
            reverse=True
        )
        
        report['ranking'] = {symbol: result for symbol, result in ranked_results[:5]}
        
        # Generate recommendations
        if ranked_results:
            best_strategy = ranked_results[0]
            report['recommendations'].append(
                f"Best performing strategy: {best_strategy[0]} (Sharpe: {best_strategy[1]['sharpe_ratio']:.2f})"
            )
        
        return report
    
    def generate_backtest_report(self, save_path=None):
        """Generate comprehensive backtest report"""
        report = {
            'backtest_date': datetime.now(),
            'initial_capital': self.initial_capital,
            'total_symbols_tested': len(self.results),
            'overall_performance': self._calculate_overall_performance(),
            'detailed_results': self.results,
            'trade_analysis': self._analyze_trades(),
            'risk_metrics': self._calculate_portfolio_risk_metrics()
        }
        
        if save_path:
            import joblib
            joblib.dump(report, save_path)
            print(f"💾 Backtest report saved to {save_path}")
        
        return report
    
    def _calculate_overall_performance(self):
        """Calculate overall backtest performance"""
        if not self.results:
            return {}
        
        returns = [result['total_return'] for result in self.results.values()]
        sharpe_ratios = [result.get('sharpe_ratio', 0) for result in self.results.values()]
        
        return {
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'return_std': np.std(returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'positive_percentage': sum(1 for r in returns if r > 0) / len(returns)
        }
    
    def _analyze_trades(self):
        """Analyze trade performance"""
        if not self.trade_log:
            return {}
        
        closed_trades = [t for t in self.trade_log if t.get('action') == 'SELL']
        
        if not closed_trades:
            return {}
        
        # Group by symbol
        symbol_trades = {}
        for trade in closed_trades:
            symbol = trade['symbol']
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)
        
        analysis = {}
        for symbol, trades in symbol_trades.items():
            pnls = [t['pnl'] for t in trades]
            analysis[symbol] = {
                'total_trades': len(trades),
                'total_pnl': sum(pnls),
                'avg_pnl': np.mean(pnls),
                'win_rate': sum(1 for p in pnls if p > 0) / len(pnls),
                'best_trade': max(pnls),
                'worst_trade': min(pnls)
            }
        
        return analysis
    
    def _calculate_portfolio_risk_metrics(self):
        """Calculate portfolio-level risk metrics"""
        # This would involve more sophisticated portfolio risk calculations
        # For now, return basic metrics from individual backtests
        return {
            'avg_max_drawdown': np.mean([r.get('max_drawdown', 0) for r in self.results.values()]),
            'avg_volatility': np.mean([r.get('volatility', 0) for r in self.results.values()])
        }

# Test function
def test_backtesting_engine():
    """Test the backtesting engine"""
    print("🧪 Testing Backtesting Engine...")
    
    engine = BacktestingEngine(initial_capital=10000)
    
    # Generate sample data for multiple symbols
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    sample_data = {
        'AAPL': pd.DataFrame({
            'Close': np.random.normal(150, 5, 100).cumsum() + 1500,
            'Open': np.random.normal(149, 5, 100).cumsum() + 1500,
            'High': np.random.normal(152, 5, 100).cumsum() + 1500,
            'Low': np.random.normal(148, 5, 100).cumsum() + 1500,
            'Volume': np.random.normal(1000000, 100000, 100)
        }, index=dates),
        'GOOGL': pd.DataFrame({
            'Close': np.random.normal(2500, 50, 100).cumsum() + 25000,
            'Open': np.random.normal(2490, 50, 100).cumsum() + 25000,
            'High': np.random.normal(2520, 50, 100).cumsum() + 25000,
            'Low': np.random.normal(2480, 50, 100).cumsum() + 25000,
            'Volume': np.random.normal(500000, 50000, 100)
        }, index=dates)
    }
    
    # Test single symbol backtest
    from trading.strategies.ml_enhanced_strategy import MLEnhancedTradingStrategy
    
    result = engine.run_backtest(sample_data['AAPL'], MLEnhancedTradingStrategy, 'AAPL')
    print(f"📈 AAPL Backtest Result: Return = {result['total_return']:.2%}")
    
    # Test comparative analysis
    comparative_results, report = engine.run_comparative_analysis(
        sample_data, MLEnhancedTradingStrategy
    )
    
    print("✅ Backtesting engine test completed!")
    return engine

if __name__ == "__main__":
    test_backtesting_engine()