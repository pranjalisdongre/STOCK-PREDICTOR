import pandas as pd
import numpy as np
import scipy.optimize as sco
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate
        self.optimization_history = []
    
    def calculate_portfolio_metrics(self, weights, expected_returns, cov_matrix):
        """Calculate portfolio performance metrics"""
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_sharpe_ratio(self, expected_returns, cov_matrix, max_allocation=0.3):
        """Optimize portfolio for maximum Sharpe ratio"""
        num_assets = len(expected_returns)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Sum to 1
        
        # Bounds (no short selling, with max allocation)
        bounds = tuple((0, max_allocation) for _ in range(num_assets))
        
        # Initial guess (equal weights)
        initial_weights = num_assets * [1.0 / num_assets]
        
        # Objective function (negative Sharpe for minimization)
        def negative_sharpe(weights):
            metrics = self.calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
            return -metrics['sharpe_ratio']
        
        # Optimization
        result = sco.minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimized_weights = result.x
            portfolio_metrics = self.calculate_portfolio_metrics(optimized_weights, expected_returns, cov_matrix)
            
            optimization_result = {
                'timestamp': datetime.now(),
                'weights': dict(zip(expected_returns.index, optimized_weights)),
                'metrics': portfolio_metrics,
                'method': 'sharpe_optimization'
            }
            
            self.optimization_history.append(optimization_result)
            return optimization_result
        
        return None
    
    def optimize_minimum_variance(self, cov_matrix, max_allocation=0.3):
        """Optimize portfolio for minimum variance"""
        num_assets = len(cov_matrix)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds
        bounds = tuple((0, max_allocation) for _ in range(num_assets))
        
        # Initial guess
        initial_weights = num_assets * [1.0 / num_assets]
        
        # Objective function (portfolio variance)
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Optimization
        result = sco.minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimized_weights = result.x
            
            # Calculate metrics
            portfolio_volatility = np.sqrt(result.fun)
            
            optimization_result = {
                'timestamp': datetime.now(),
                'weights': dict(zip(cov_matrix.index, optimized_weights)),
                'metrics': {
                    'volatility': portfolio_volatility,
                    'method': 'minimum_variance'
                }
            }
            
            self.optimization_history.append(optimization_result)
            return optimization_result
        
        return None
    
    def black_litterman_optimization(self, historical_returns, views, view_confidences, tau=0.05):
        """Black-Litterman portfolio optimization"""
        # Implied equilibrium returns
        pi = self._calculate_implied_returns(historical_returns)
        
        # Views matrix
        P, Q = self._create_views_matrix(views, historical_returns.columns)
        
        # Uncertainty matrix
        omega = self._create_uncertainty_matrix(view_confidences, historical_returns.cov())
        
        # Black-Litterman formula
        cov_matrix = historical_returns.cov()
        M = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + P.T @ np.linalg.inv(omega) @ P)
        bl_returns = M @ (np.linalg.inv(tau * cov_matrix) @ pi + P.T @ np.linalg.inv(omega) @ Q)
        
        # Convert to expected returns series
        bl_expected_returns = pd.Series(bl_returns, index=historical_returns.columns)
        
        return bl_expected_returns
    
    def _calculate_implied_returns(self, historical_returns, delta=2.5):
        """Calculate implied equilibrium returns"""
        cov_matrix = historical_returns.cov()
        market_caps = self._estimate_market_caps(historical_returns.columns)
        market_weights = market_caps / market_caps.sum()
        
        # Implied returns
        pi = delta * cov_matrix @ market_weights
        
        return pd.Series(pi, index=historical_returns.columns)
    
    def _estimate_market_caps(self, symbols):
        """Estimate market capitalizations (simplified)"""
        # In production, use actual market cap data
        market_caps = {}
        for symbol in symbols:
            if symbol == 'AAPL': market_caps[symbol] = 2.5e12
            elif symbol == 'GOOGL': market_caps[symbol] = 1.8e12
            elif symbol == 'MSFT': market_caps[symbol] = 2.0e12
            elif symbol == 'AMZN': market_caps[symbol] = 1.5e12
            elif symbol == 'TSLA': market_caps[symbol] = 0.8e12
            else: market_caps[symbol] = 0.5e12  # Default
        
        return pd.Series(market_caps)
    
    def _create_views_matrix(self, views, symbols):
        """Create views matrix for Black-Litterman"""
        num_views = len(views)
        num_assets = len(symbols)
        
        P = np.zeros((num_views, num_assets))
        Q = np.zeros(num_views)
        
        for i, (view_assets, view_return) in enumerate(views.items()):
            if isinstance(view_assets, str):
                # Absolute view on single asset
                asset_idx = symbols.get_loc(view_assets)
                P[i, asset_idx] = 1
            else:
                # Relative view between assets
                for asset, weight in view_assets.items():
                    asset_idx = symbols.get_loc(asset)
                    P[i, asset_idx] = weight
            
            Q[i] = view_return
        
        return P, Q
    
    def _create_uncertainty_matrix(self, view_confidences, cov_matrix):
        """Create uncertainty matrix for views"""
        num_views = len(view_confidences)
        omega = np.zeros((num_views, num_views))
        
        for i, confidence in enumerate(view_confidences):
            omega[i, i] = 1 / confidence  # Higher confidence = lower uncertainty
        
        return omega
    
    def calculate_efficient_frontier(self, expected_returns, cov_matrix, num_portfolios=1000):
        """Calculate efficient frontier"""
        num_assets = len(expected_returns)
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            results[0,i] = portfolio_volatility
            results[1,i] = portfolio_return
            results[2,i] = sharpe_ratio
            weights_record.append(weights)
        
        return results, weights_record
    
    def rebalance_portfolio(self, current_weights, target_weights, prices, transaction_cost=0.001):
        """Calculate rebalancing trades with transaction costs"""
        current_value = sum(current_weights.values())
        target_values = {symbol: weight * current_value for symbol, weight in target_weights.items()}
        
        trades = {}
        total_transaction_cost = 0
        
        for symbol in set(current_weights.keys()) | set(target_weights.keys()):
            current_val = current_weights.get(symbol, 0) * current_value
            target_val = target_values.get(symbol, 0)
            
            trade_amount = target_val - current_val
            
            if abs(trade_amount) > 0:
                # Calculate shares to trade
                shares = trade_amount / prices.get(symbol, 1)
                transaction_cost_amount = abs(trade_amount) * transaction_cost
                
                trades[symbol] = {
                    'shares': shares,
                    'amount': trade_amount,
                    'transaction_cost': transaction_cost_amount,
                    'action': 'BUY' if trade_amount > 0 else 'SELL'
                }
                
                total_transaction_cost += transaction_cost_amount
        
        return {
            'trades': trades,
            'total_transaction_cost': total_transaction_cost,
            'rebalancing_cost_percent': total_transaction_cost / current_value
        }

# Test function
def test_portfolio_optimizer():
    """Test the portfolio optimizer"""
    print("🧪 Testing Portfolio Optimizer...")
    
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Sample data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Expected returns (annualized)
    expected_returns = pd.Series({
        'AAPL': 0.12,
        'GOOGL': 0.15,
        'MSFT': 0.10,
        'AMZN': 0.18,
        'TSLA': 0.25
    })
    
    # Covariance matrix
    cov_matrix = pd.DataFrame({
        'AAPL': [0.04, 0.02, 0.01, 0.03, 0.05],
        'GOOGL': [0.02, 0.06, 0.02, 0.04, 0.06],
        'MSFT': [0.01, 0.02, 0.03, 0.02, 0.04],
        'AMZN': [0.03, 0.04, 0.02, 0.08, 0.07],
        'TSLA': [0.05, 0.06, 0.04, 0.07, 0.15]
    }, index=symbols)
    
    # Test Sharpe ratio optimization
    result = optimizer.optimize_sharpe_ratio(expected_returns, cov_matrix)
    print(f"📊 Sharpe Optimization Result:")
    print(f"   Weights: {result['weights']}")
    print(f"   Sharpe Ratio: {result['metrics']['sharpe_ratio']:.3f}")
    
    # Test minimum variance optimization
    min_var_result = optimizer.optimize_minimum_variance(cov_matrix)
    print(f"📈 Minimum Variance Result:")
    print(f"   Weights: {min_var_result['weights']}")
    print(f"   Volatility: {min_var_result['metrics']['volatility']:.3f}")
    
    print("✅ Portfolio optimizer test completed!")
    return optimizer

if __name__ == "__main__":
    test_portfolio_optimizer()