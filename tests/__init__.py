"""
Test Suite for AI Stock Predictor
=================================

Comprehensive test coverage for all modules and components.
"""

import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

__version__ = "1.0.0"
__all__ = [
    'test_data_collectors',
    'test_ml_models', 
    'test_trading_strategies',
    'test_web_app',
    'test_utils'
]