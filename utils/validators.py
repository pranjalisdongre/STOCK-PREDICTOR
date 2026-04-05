import re
from datetime import datetime, timedelta
from typing import Union, Tuple
import pandas as pd
from config.settings import Config

def validate_symbol(symbol: str) -> Tuple[bool, str]:
    """
    Validate stock symbol format.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not symbol or not isinstance(symbol, str):
        return False, "Symbol must be a non-empty string"
    
    # Basic format check (1-5 uppercase letters)
    if not re.match(r'^[A-Z]{1,5}$', symbol.upper()):
        return False, "Symbol must be 1-5 uppercase letters"
    
    # Check if symbol is in our allowed list
    if symbol.upper() not in Config.DEFAULT_SYMBOLS:
        return False, f"Symbol {symbol} not in configured symbol list"
    
    return True, "Valid symbol"

def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, str]:
    """
    Validate date range for data queries.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start > end:
            return False, "Start date cannot be after end date"
        
        if start > datetime.now():
            return False, "Start date cannot be in the future"
        
        # Maximum 5 years of historical data
        max_days = 5 * 365
        if (end - start).days > max_days:
            return False, f"Date range cannot exceed {max_days} days"
        
        return True, "Valid date range"
        
    except ValueError as e:
        return False, f"Invalid date format: {e}"

def validate_price(price: Union[float, int]) -> Tuple[bool, str]:
    """
    Validate stock price.
    
    Args:
        price: Price to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(price, (int, float)):
        return False, "Price must be a number"
    
    if price <= 0:
        return False, "Price must be positive"
    
    if price > 1000000:  # $1 million upper limit
        return False, "Price appears to be unrealistically high"
    
    return True, "Valid price"

def validate_quantity(quantity: Union[int, float]) -> Tuple[bool, str]:
    """
    Validate trade quantity.
    
    Args:
        quantity: Quantity to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(quantity, (int, float)):
        return False, "Quantity must be a number"
    
    if quantity <= 0:
        return False, "Quantity must be positive"
    
    if not quantity.is_integer() if isinstance(quantity, float) else True:
        return False, "Quantity must be a whole number for stocks"
    
    if quantity > 100000:  # Reasonable upper limit
        return False, "Quantity appears to be unrealistically high"
    
    return True, "Valid quantity"

def validate_portfolio_allocation(allocations: dict) -> Tuple[bool, str]:
    """
    Validate portfolio allocation weights.
    
    Args:
        allocations: Dictionary of symbol -> weight
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not allocations:
        return False, "Allocations cannot be empty"
    
    total_weight = sum(allocations.values())
    
    # Check if weights sum to approximately 1 (allowing for floating point errors)
    if abs(total_weight - 1.0) > 0.01:
        return False, f"Allocations must sum to 1.0 (current sum: {total_weight})"
    
    # Check individual weights
    for symbol, weight in allocations.items():
        is_valid, message = validate_symbol(symbol)
        if not is_valid:
            return False, f"Invalid symbol {symbol}: {message}"
        
        if weight < 0:
            return False, f"Weight for {symbol} cannot be negative"
        
        if weight > Config.MAX_ALLOCATION_PER_ASSET:
            return False, f"Weight for {symbol} exceeds maximum allocation"
    
    return True, "Valid portfolio allocation"

def validate_risk_parameters(risk_per_trade: float, max_drawdown: float) -> Tuple[bool, str]:
    """
    Validate risk management parameters.
    
    Args:
        risk_per_trade: Risk per trade as decimal
        max_drawdown: Maximum drawdown as decimal
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not 0 < risk_per_trade <= 0.1:  # Max 10% risk per trade
        return False, "Risk per trade must be between 0 and 0.1"
    
    if not 0 < max_drawdown <= 0.5:  # Max 50% drawdown
        return False, "Max drawdown must be between 0 and 0.5"
    
    if risk_per_trade > max_drawdown:
        return False, "Risk per trade cannot exceed max drawdown"
    
    return True, "Valid risk parameters"

def validate_ml_prediction(prediction: dict) -> Tuple[bool, str]:
    """
    Validate ML prediction structure.
    
    Args:
        prediction: Prediction dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ['ensemble_prediction', 'confidence']
    
    for field in required_fields:
        if field not in prediction:
            return False, f"Missing required field: {field}"
    
    if not isinstance(prediction['confidence'], (int, float)):
        return False, "Confidence must be a number"
    
    if not 0 <= prediction['confidence'] <= 1:
        return False, "Confidence must be between 0 and 1"
    
    return True, "Valid prediction"