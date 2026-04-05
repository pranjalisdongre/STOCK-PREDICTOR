import logging
import time
import functools
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from typing import Any, Callable, Optional
import requests
from config.settings import Config

def setup_logging(name: str = None, level: str = None) -> logging.Logger:
    """
    Set up logging with consistent format.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    if level is None:
        level = Config.LOG_LEVEL
    
    if name is None:
        name = 'stock_predictor'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(Config.LOG_FORMAT)
    
    # File handler
    file_handler = logging.FileHandler(f'logs/{Config.LOG_FILE}')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = setup_logging('timer')
        logger.debug(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0, 
          exceptions: tuple = (Exception,)) -> Callable:
    """
    Retry decorator for handling transient failures.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds
        exceptions: Exceptions to catch and retry on
        
    Returns:
        Wrapped function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger = setup_logging('retry')
                    
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        return wrapper
    return decorator

def cache_result(ttl: int = 300) -> Callable:
    """
    Simple in-memory cache decorator with TTL.
    
    Args:
        ttl: Time to live in seconds
        
    Returns:
        Wrapped function
    """
    cache = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check if result is in cache and not expired
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            return result
        return wrapper
    return decorator

def is_market_open() -> bool:
    """
    Check if US stock market is currently open.
    
    Returns:
        True if market is open, False otherwise
    """
    now = datetime.now()
    
    # Check if weekend
    if now.weekday() >= 5:  # Saturday (5) or Sunday (6)
        return False
    
    # Check time (Eastern Time)
    from config.constants import MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE
    
    # Simple check - in production, use market calendar
    current_time = now.time()
    market_open = datetime.now().replace(
        hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0
    ).time()
    market_close = datetime.now().replace(
        hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0
    ).time()
    
    return market_open <= current_time <= market_close

def get_next_market_open() -> datetime:
    """
    Get datetime of next market open.
    
    Returns:
        Datetime of next market open
    """
    now = datetime.now()
    
    # If it's weekend, next open is Monday 9:30 AM
    if now.weekday() >= 5:
        days_until_monday = (7 - now.weekday()) % 7
        next_open = now + timedelta(days=days_until_monday)
    else:
        # If market is closed for the day, next open is tomorrow
        if not is_market_open():
            next_open = now + timedelta(days=1)
        else:
            next_open = now
    
    # Set to market open time
    from config.constants import MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE
    next_open = next_open.replace(
        hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0
    )
    
    return next_open

def download_file(url: str, filepath: str, timeout: int = 30) -> bool:
    """
    Download file from URL with error handling.
    
    Args:
        url: URL to download from
        filepath: Local file path to save to
        timeout: Request timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
        
    except Exception as e:
        logger = setup_logging('download')
        logger.error(f"Failed to download {url}: {e}")
        return False

def create_directory(path: str) -> bool:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path
        
    Returns:
        True if successful or already exists, False otherwise
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger = setup_logging('directory')
        logger.error(f"Failed to create directory {path}: {e}")
        return False

def memory_usage() -> dict:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory usage details
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'error': 'psutil not installed'}

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame by handling missing values and outliers.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Forward fill then backward fill missing values
    df_clean = df_clean.ffill().bfill()
    
    # Remove any remaining NaN rows
    df_clean = df_clean.dropna()
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    return df_clean

@timer
def batch_process(data: list, batch_size: int, process_func: Callable) -> list:
    """
    Process data in batches.
    
    Args:
        data: List of data to process
        batch_size: Size of each batch
        process_func: Function to process each batch
        
    Returns:
        Combined results from all batches
    """
    results = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
    
    return results