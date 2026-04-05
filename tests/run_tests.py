#!/usr/bin/env python3
"""
Test Runner for AI Stock Predictor
==================================

Run all tests or specific test modules.
"""

import pytest
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_all_tests():
    """Run all tests."""
    print("🚀 Running all tests for AI Stock Predictor...")
    return pytest.main([
        'tests/', '-v', '--tb=short', '--color=yes'
    ])


def run_unit_tests():
    """Run only unit tests."""
    print("🧪 Running unit tests...")
    return pytest.main([
        'tests/', '-v', '--tb=short', '-m', 'not slow and not integration'
    ])


def run_integration_tests():
    """Run only integration tests."""
    print("🔗 Running integration tests...")
    return pytest.main([
        'tests/', '-v', '--tb=short', '-m', 'integration'
    ])


def run_web_tests():
    """Run only web application tests."""
    print("🌐 Running web tests...")
    return pytest.main([
        'tests/test_web_app.py', '-v', '--tb=short'
    ])


def run_with_coverage():
    """Run tests with coverage report."""
    print("📊 Running tests with coverage...")
    return pytest.main([
        'tests/', '-v', '--tb=short', '--cov=.', '--cov-report=html'
    ])


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AI Stock Predictor tests')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--web', action='store_true', help='Run only web tests')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage')
    parser.add_argument('--slow', action='store_true', help='Include slow tests')
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = ['tests/', '-v', '--tb=short', '--color=yes']
    
    if args.unit:
        pytest_args.extend(['-m', 'not integration and not web'])
    elif args.integration:
        pytest_args.extend(['-m', 'integration'])
    elif args.web:
        pytest_args.extend(['-m', 'web'])
    
    if args.coverage:
        pytest_args.extend(['--cov=.', '--cov-report=html'])
    
    if args.slow:
        pytest_args.append('--run-slow')
    else:
        pytest_args.extend(['-m', 'not slow'])
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)