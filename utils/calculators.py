import numpy as np
import pandas as pd
from typing import List, Dict, Union
from config.constants import TRADING_DAYS_PER_YEAR

def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate percentage returns from price series.
    
    Args:
        prices: Series of prices
        
    Returns:
        Series of percentage returns
    """
    return prices.pct_change().dropna()

def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Calculate volatility from returns.
    
    Args:
        returns: Series of returns
        annualize: Whether to annualize the volatility
        
    Returns:
        Volatility as float
    """
    if returns.empty:
        return 0.0
    
    volatility = returns.std()
    
    if annualize:
        volatility *= np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return volatility

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio from returns.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    
    # Annualize returns and risk-free rate
    annual_return = returns.mean() * TRADING_DAYS_PER_YEAR
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return (annual_return - risk_free_rate) / annual_volatility

def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown from price series.
    
    Args:
        prices: Series of prices
        
    Returns:
        Maximum drawdown as decimal
    """
    if prices.empty:
        return 0.0
    
    cumulative_returns = (1 + calculate_returns(prices)).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    
    return drawdown.min()

def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta (market sensitivity) of an asset.
    
    Args:
        asset_returns: Returns of the asset
        market_returns: Returns of the market benchmark
        
    Returns:
        Beta coefficient
    """
    if len(asset_returns) != len(market_returns) or asset_returns.empty:
        return 1.0  # Default to market beta
    
    # Align the series
    aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if aligned_data.empty:
        return 1.0
    
    asset_aligned = aligned_data.iloc[:, 0]
    market_aligned = aligned_data.iloc[:, 1]
    
    covariance = asset_aligned.cov(market_aligned)
    market_variance = market_aligned.var()
    
    if market_variance == 0:
        return 1.0
    
    return covariance / market_variance

def calculate_correlation(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calculate correlation between two series.
    
    Args:
        series1: First series
        series2: Second series
        
    Returns:
        Correlation coefficient
    """
    if series1.empty or series2.empty:
        return 0.0
    
    aligned_data = pd.concat([series1, series2], axis=1).dropna()
    if aligned_data.empty:
        return 0.0
    
    return aligned_data.corr().iloc[0, 1]

def calculate_portfolio_variance(weights: List[float], cov_matrix: pd.DataFrame) -> float:
    """
    Calculate portfolio variance.
    
    Args:
        weights: List of asset weights
        cov_matrix: Covariance matrix of asset returns
        
    Returns:
        Portfolio variance
    """
    weights_array = np.array(weights)
    return np.dot(weights_array.T, np.dot(cov_matrix, weights_array))

def calculate_portfolio_return(weights: List[float], expected_returns: List[float]) -> float:
    """
    Calculate expected portfolio return.
    
    Args:
        weights: List of asset weights
        expected_returns: List of expected returns
        
    Returns:
        Expected portfolio return
    """
    return np.dot(weights, expected_returns)

def calculate_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using historical method.
    
    Args:
        returns: Series of returns
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Value at Risk as decimal
    """
    if returns.empty:
        return 0.0
    
    return returns.quantile(1 - confidence)

def calculate_expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Expected Shortfall (CVaR) using historical method.
    
    Args:
        returns: Series of returns
        confidence: Confidence level
        
    Returns:
        Expected Shortfall as decimal
    """
    if returns.empty:
        return 0.0
    
    var = calculate_value_at_risk(returns, confidence)
    tail_returns = returns[returns <= var]
    
    if tail_returns.empty:
        return var
    
    return tail_returns.mean()

def calculate_trade_pnl(entry_price: float, exit_price: float, quantity: int, 
                       commission: float = 0.001) -> Dict[str, float]:
    """
    Calculate trade profit and loss.
    
    Args:
        entry_price: Entry price per share
        exit_price: Exit price per share  
        quantity: Number of shares
        commission: Commission rate as decimal
        
    Returns:
        Dictionary with P&L details
    """
    gross_pnl = (exit_price - entry_price) * quantity
    entry_commission = entry_price * quantity * commission
    exit_commission = exit_price * quantity * commission
    total_commission = entry_commission + exit_commission
    net_pnl = gross_pnl - total_commission
    pnl_percent = (exit_price - entry_price) / entry_price
    
    return {
        'gross_pnl': gross_pnl,
        'total_commission': total_commission,
        'net_pnl': net_pnl,
        'pnl_percent': pnl_percent,
        'entry_commission': entry_commission,
        'exit_commission': exit_commission
    }

def calculate_position_size(portfolio_value: float, entry_price: float, 
                          stop_loss_price: float, risk_per_trade: float = 0.02) -> int:
    """
    Calculate position size based on risk management.
    
    Args:
        portfolio_value: Total portfolio value
        entry_price: Entry price per share
        stop_loss_price: Stop loss price per share
        risk_per_trade: Risk per trade as decimal
        
    Returns:
        Number of shares to trade
    """
    risk_amount = portfolio_value * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price)
    
    if price_risk == 0:
        return 0
    
    position_size = risk_amount / price_risk
    return int(position_size)