from typing import Union
import pandas as pd
from datetime import datetime

def format_currency(amount: Union[float, int], include_symbol: bool = True) -> str:
    """
    Format number as currency.
    
    Args:
        amount: Amount to format
        include_symbol: Whether to include currency symbol
        
    Returns:
        Formatted currency string
    """
    if amount is None:
        return "N/A"
    
    symbol = "$" if include_symbol else ""
    
    if abs(amount) >= 1_000_000:
        return f"{symbol}{amount/1_000_000:.2f}M"
    elif abs(amount) >= 1_000:
        return f"{symbol}{amount/1_000:.2f}K"
    else:
        return f"{symbol}{amount:.2f}"

def format_percentage(value: float, decimals: int = 2, include_symbol: bool = True) -> str:
    """
    Format number as percentage.
    
    Args:
        value: Value to format (as decimal, e.g., 0.05 for 5%)
        decimals: Number of decimal places
        include_symbol: Whether to include percentage symbol
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"
    
    symbol = "%" if include_symbol else ""
    formatted = f"{value * 100:.{decimals}f}{symbol}"
    
    # Add color indication for positive/negative
    if value > 0 and include_symbol:
        formatted = f"+{formatted}"
    
    return formatted

def format_large_number(number: Union[int, float]) -> str:
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        number: Number to format
        
    Returns:
        Formatted number string
    """
    if number is None:
        return "N/A"
    
    if abs(number) >= 1_000_000_000:
        return f"{number/1_000_000_000:.2f}B"
    elif abs(number) >= 1_000_000:
        return f"{number/1_000_000:.2f}M"
    elif abs(number) >= 1_000:
        return f"{number/1_000:.2f}K"
    else:
        return f"{number:.2f}"

def format_timestamp(timestamp: Union[str, datetime], format: str = None) -> str:
    """
    Format timestamp to readable string.
    
    Args:
        timestamp: Timestamp to format
        format: strftime format string
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        return "N/A"
    
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return timestamp
    
    if format is None:
        format = "%Y-%m-%d %H:%M:%S"
    
    return timestamp.strftime(format)

def format_trade_signal(signal: str) -> str:
    """
    Format trading signal for display.
    
    Args:
        signal: Raw signal string
        
    Returns:
        Formatted signal string
    """
    signal_map = {
        'BUY': '🟢 BUY',
        'SELL': '🔴 SELL', 
        'HOLD': '🟡 HOLD',
        'WEAK_BUY': '🟢 WEAK BUY',
        'WEAK_SELL': '🔴 WEAK SELL'
    }
    
    return signal_map.get(signal, f'⚪ {signal}')

def format_confidence(confidence: float) -> str:
    """
    Format confidence level with color indication.
    
    Args:
        confidence: Confidence level (0-1)
        
    Returns:
        Formatted confidence string
    """
    if confidence is None:
        return "N/A"
    
    if confidence >= 0.8:
        color = "🟢"  # High confidence
    elif confidence >= 0.6:
        color = "🟡"  # Medium confidence
    else:
        color = "🔴"  # Low confidence
    
    return f"{color} {confidence:.1%}"

def format_volatility(volatility: float) -> str:
    """
    Format volatility for display.
    
    Args:
        volatility: Annualized volatility
        
    Returns:
        Formatted volatility string
    """
    if volatility is None:
        return "N/A"
    
    if volatility >= 0.4:
        level = "🔴 HIGH"
    elif volatility >= 0.2:
        level = "🟡 MEDIUM" 
    else:
        level = "🟢 LOW"
    
    return f"{level} ({volatility:.1%})"

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds is None:
        return "N/A"
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"

def format_dataframe_for_display(df: pd.DataFrame, max_rows: int = 10) -> str:
    """
    Format DataFrame for console display.
    
    Args:
        df: DataFrame to format
        max_rows: Maximum number of rows to display
        
    Returns:
        Formatted string representation
    """
    if df.empty:
        return "Empty DataFrame"
    
    # Select subset of rows
    display_df = df.head(max_rows)
    
    # Format numeric columns
    for col in display_df.select_dtypes(include=['float64']).columns:
        if any(keyword in col.lower() for keyword in ['price', 'value', 'amount']):
            display_df[col] = display_df[col].apply(lambda x: format_currency(x))
        elif any(keyword in col.lower() for keyword in ['return', 'change', 'percent']):
            display_df[col] = display_df[col].apply(lambda x: format_percentage(x))
    
    return display_df.to_string()