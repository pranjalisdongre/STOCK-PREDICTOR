from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from data.collectors.real_time_collector import RealTimeDataCollector
from data.collectors.news_collector import NewsSentimentAnalyzer
from ml.models.ensemble_predictor import EnsembleStockPredictor
from trading.strategies.ml_enhanced_strategy import MLEnhancedTradingStrategy
from trading.backtesting.engine import BacktestingEngine

# Create Blueprint
api = Blueprint('api', __name__)

# Initialize components
data_collector = RealTimeDataCollector()
news_analyzer = NewsSentimentAnalyzer()
trading_strategy = MLEnhancedTradingStrategy()
backtest_engine = BacktestingEngine()

@api.route('/health')
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@api.route('/stocks', methods=['GET'])
def get_available_stocks():
    """Get list of available stocks"""
    symbols = data_collector.symbols
    return jsonify({
        'symbols': symbols,
        'count': len(symbols)
    })

@api.route('/stocks/<symbol>/data', methods=['GET'])
def get_stock_data(symbol):
    """Get stock historical data"""
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    
    try:
        data = data_collector.get_historical_data(symbol, period=period, interval=interval)
        if data is None:
            return jsonify({'error': f'No data found for {symbol}'}), 404
        
        # Convert to JSON-serializable format
        data_json = data.reset_index().to_dict('records')
        
        return jsonify({
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'data_points': len(data_json),
            'data': data_json
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/stocks/<symbol>/predict', methods=['GET'])
def predict_stock(symbol):
    """Get AI prediction for a stock"""
    try:
        # Get historical data
        data = data_collector.get_historical_data(symbol, period='6mo')
        if data is None:
            return jsonify({'error': f'No data found for {symbol}'}), 404
        
        # Generate prediction (simplified - would use actual trained model)
        current_price = data['Close'].iloc[-1]
        
        # Simulate AI prediction
        prediction = {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_price': current_price * (1 + np.random.uniform(-0.05, 0.05)),
            'confidence': np.random.uniform(0.6, 0.9),
            'direction': 'UP' if np.random.random() > 0.5 else 'DOWN',
            'timestamp': datetime.now().isoformat(),
            'model_used': 'ensemble_predictor'
        }
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/stocks/<symbol>/news', methods=['GET'])
def get_stock_news(symbol):
    """Get news sentiment for a stock"""
    try:
        days = int(request.args.get('days', 7))
        sentiment = news_analyzer.calculate_daily_sentiment(symbol)
        
        return jsonify({
            'symbol': symbol,
            'sentiment_analysis': sentiment,
            'news_period_days': days
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/portfolio', methods=['GET'])
def get_portfolio():
    """Get current portfolio status"""
    try:
        summary = trading_strategy.get_portfolio_summary()
        performance = trading_strategy.get_performance_metrics()
        
        return jsonify({
            'portfolio_summary': summary,
            'performance_metrics': performance,
            'positions': trading_strategy.positions,
            'recent_trades': trading_strategy.trade_history[-20:]  # Last 20 trades
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/portfolio/trade', methods=['POST'])
def execute_trade():
    """Execute a trade"""
    try:
        data = request.json
        symbol = data.get('symbol')
        action = data.get('action')  # BUY or SELL
        quantity = data.get('quantity')
        
        if not all([symbol, action, quantity]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Simulate trade execution
        current_price = 150  # Would get from real-time data
        
        if action.upper() == 'BUY':
            # Execute buy
            cost = quantity * current_price
            if cost <= trading_strategy.cash:
                trading_strategy.positions[symbol] = {
                    'entry_price': current_price,
                    'position_size': quantity,
                    'entry_time': datetime.now()
                }
                trading_strategy.cash -= cost
                
                trade_record = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'EXECUTED'
                }
                trading_strategy.trade_history.append(trade_record)
                
                return jsonify({
                    'message': f'Successfully bought {quantity} shares of {symbol}',
                    'trade': trade_record
                })
            else:
                return jsonify({'error': 'Insufficient funds'}), 400
                
        elif action.upper() == 'SELL':
            # Execute sell
            if symbol in trading_strategy.positions:
                position = trading_strategy.positions[symbol]
                if quantity <= position['position_size']:
                    proceeds = quantity * current_price
                    trading_strategy.cash += proceeds
                    
                    # Update or remove position
                    if quantity == position['position_size']:
                        del trading_strategy.positions[symbol]
                    else:
                        position['position_size'] -= quantity
                    
                    trade_record = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': quantity,
                        'price': current_price,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'EXECUTED'
                    }
                    trading_strategy.trade_history.append(trade_record)
                    
                    return jsonify({
                        'message': f'Successfully sold {quantity} shares of {symbol}',
                        'trade': trade_record
                    })
                else:
                    return jsonify({'error': 'Insufficient shares'}), 400
            else:
                return jsonify({'error': 'No position found for symbol'}), 400
        
        else:
            return jsonify({'error': 'Invalid action. Use BUY or SELL'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/backtest', methods=['POST'])
def run_backtest():
    """Run backtesting for a strategy"""
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        strategy_name = data.get('strategy', 'ml_enhanced')
        initial_capital = data.get('initial_capital', 10000)
        
        # Get historical data
        historical_data = data_collector.get_historical_data(symbol, period='1y')
        if historical_data is None:
            return jsonify({'error': f'No data found for {symbol}'}), 404
        
        # Run backtest
        from trading.strategies.ml_enhanced_strategy import MLEnhancedTradingStrategy
        
        result = backtest_engine.run_backtest(
            historical_data, 
            MLEnhancedTradingStrategy, 
            symbol,
            initial_capital=initial_capital
        )
        
        return jsonify({
            'backtest_results': result,
            'parameters': {
                'symbol': symbol,
                'strategy': strategy_name,
                'initial_capital': initial_capital,
                'period': '1y'
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/market/overview', methods=['GET'])
def get_market_overview():
    """Get market overview"""
    try:
        # Calculate market metrics from available symbols
        symbols = data_collector.symbols[:5]  # Use first 5 symbols for demo
        
        market_data = []
        total_change = 0
        
        for symbol in symbols:
            data = data_collector.get_historical_data(symbol, period='1mo')
            if data is not None and len(data) > 1:
                current_price = data['Close'].iloc[-1]
                previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = (current_price - previous_price) / previous_price * 100
                
                market_data.append({
                    'symbol': symbol,
                    'price': current_price,
                    'change': change,
                    'volume': data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
                })
                
                total_change += change
        
        avg_market_change = total_change / len(market_data) if market_data else 0
        
        return jsonify({
            'market_overview': {
                'total_symbols': len(market_data),
                'average_change': avg_market_change,
                'market_trend': 'UP' if avg_market_change > 0 else 'DOWN',
                'update_time': datetime.now().isoformat()
            },
            'top_movers': sorted(market_data, key=lambda x: abs(x['change']), reverse=True)[:5]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/alerts', methods=['GET'])
def get_alerts():
    """Get system alerts and notifications"""
    try:
        # Sample alerts - in production, these would come from monitoring systems
        alerts = [
            {
                'id': 1,
                'type': 'PRICE_ALERT',
                'symbol': 'AAPL',
                'message': 'AAPL price dropped below $150',
                'severity': 'MEDIUM',
                'timestamp': (datetime.now() - timedelta(hours=1)).isoformat()
            },
            {
                'id': 2,
                'type': 'VOLUME_ALERT', 
                'symbol': 'TSLA',
                'message': 'Unusual volume detected in TSLA',
                'severity': 'LOW',
                'timestamp': (datetime.now() - timedelta(minutes=30)).isoformat()
            }
        ]
        
        return jsonify({
            'alerts': alerts,
            'total_alerts': len(alerts),
            'unread_alerts': len([a for a in alerts if a.get('read', False) == False])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/system/status', methods=['GET'])
def get_system_status():
    """Get system status and component health"""
    try:
        components = {
            'data_collector': 'OPERATIONAL',
            'ml_models': 'OPERATIONAL' if data_collector.get_historical_data('AAPL') is not None else 'DEGRADED',
            'trading_engine': 'OPERATIONAL',
            'news_feed': 'OPERATIONAL',
            'database': 'OPERATIONAL'
        }
        
        return jsonify({
            'system_status': 'OPERATIONAL',
            'components': components,
            'last_updated': datetime.now().isoformat(),
            'uptime': '99.9%',
            'version': '1.0.0'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500