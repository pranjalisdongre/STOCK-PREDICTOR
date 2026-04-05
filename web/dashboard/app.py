from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Import our project modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from data.collectors.real_time_collector import RealTimeDataCollector
from data.collectors.news_collector import NewsSentimentAnalyzer
from ml.models.ensemble_predictor import EnsembleStockPredictor
from trading.strategies.ml_enhanced_strategy import MLEnhancedTradingStrategy
from trading.strategies.risk_manager import RiskManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

class DashboardManager:
    def __init__(self):
        self.data_collector = RealTimeDataCollector()
        self.news_analyzer = NewsSentimentAnalyzer()
        self.trading_strategy = MLEnhancedTradingStrategy()
        self.risk_manager = RiskManager()
        self.predictor = None
        self.live_data = {}
        self.portfolio_data = {}
        self.alerts = []
        
        # Load sample models (in production, load trained models)
        try:
            self.predictor = EnsembleStockPredictor()
            print("✅ ML models initialized")
        except Exception as e:
            print(f"❌ Error loading ML models: {e}")
    
    def get_market_overview(self):
        """Get overall market overview"""
        overview = {
            'timestamp': datetime.now(),
            'total_symbols': len(self.data_collector.symbols),
            'market_status': 'OPEN' if 9 <= datetime.now().hour < 16 else 'CLOSED',
            'total_portfolio_value': self.trading_strategy.portfolio_value,
            'active_positions': len(self.trading_strategy.positions),
            'today_performance': self._get_today_performance(),
            'market_sentiment': self._get_market_sentiment()
        }
        return overview
    
    def _get_today_performance(self):
        """Calculate today's portfolio performance"""
        # Simplified - in production, calculate from actual trades
        return np.random.uniform(-2, 2)  # Random between -2% to +2%
    
    def _get_market_sentiment(self):
        """Get overall market sentiment"""
        return {
            'score': np.random.uniform(-1, 1),
            'trend': 'BULLISH' if np.random.random() > 0.5 else 'BEARISH',
            'confidence': np.random.uniform(0.6, 0.9)
        }
    
    def generate_stock_charts(self, symbol, period='1mo'):
        """Generate interactive charts for a stock"""
        # Get historical data
        historical_data = self.data_collector.get_historical_data(symbol, period=period)
        
        if historical_data is None or historical_data.empty:
            return None
        
        # Create main price chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{symbol} Price Chart', 'Technical Indicators'),
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Price data
        fig.add_trace(
            go.Candlestick(
                x=historical_data.index,
                open=historical_data['Open'],
                high=historical_data['High'],
                low=historical_data['Low'],
                close=historical_data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if len(historical_data) > 20:
            historical_data['SMA_20'] = historical_data['Close'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['SMA_20'], 
                          name='SMA 20', line=dict(color='orange')),
                row=1, col=1
            )
        
        # Volume
        fig.add_trace(
            go.Bar(x=historical_data.index, y=historical_data['Volume'], 
                  name='Volume', marker_color='lightblue', opacity=0.6),
            row=2, col=1
        )
        
        # RSI (simplified)
        if len(historical_data) > 14:
            delta = historical_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            historical_data['RSI'] = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['RSI'], 
                          name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(
            height=600,
            title=f"{symbol} Technical Analysis",
            xaxis_rangeslider_visible=False
        )
        
        return fig.to_json()
    
    def get_portfolio_metrics(self):
        """Get portfolio performance metrics"""
        portfolio_summary = self.trading_strategy.get_portfolio_summary()
        performance_metrics = self.trading_strategy.get_performance_metrics()
        
        return {
            'summary': portfolio_summary,
            'performance': performance_metrics,
            'positions': self.trading_strategy.positions,
            'recent_trades': self.trading_strategy.trade_history[-10:]  # Last 10 trades
        }
    
    def generate_ai_predictions(self, symbol):
        """Generate AI predictions for a symbol"""
        try:
            # Get recent data
            historical_data = self.data_collector.get_historical_data(symbol, period='3mo')
            
            if historical_data is None or len(historical_data) < 30:
                return None
            
            # Prepare features (simplified)
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Generate prediction
            if self.predictor and self.predictor.is_trained:
                prediction = self.predictor.predict(historical_data.tail(30), feature_columns)
            else:
                # Fallback to simple prediction
                current_price = historical_data['Close'].iloc[-1]
                prediction = {
                    'ensemble_prediction': current_price * (1 + np.random.uniform(-0.02, 0.02)),
                    'confidence': np.random.uniform(0.6, 0.9),
                    'individual_predictions': {
                        'random_forest': current_price * (1 + np.random.uniform(-0.01, 0.01)),
                        'gradient_boosting': current_price * (1 + np.random.uniform(-0.01, 0.01)),
                        'lstm': current_price * (1 + np.random.uniform(-0.01, 0.01))
                    }
                }
            
            # Get news sentiment
            sentiment = self.news_analyzer.calculate_daily_sentiment(symbol)
            
            # Generate trading signal
            technical_indicators = {
                'RSI_14': np.random.uniform(20, 80),
                'MACD': np.random.uniform(-2, 2),
                'Price_vs_SMA20': np.random.uniform(-0.05, 0.05)
            }
            
            signal = self.trading_strategy.generate_signals(
                historical_data.tail(10), prediction, technical_indicators
            )
            
            return {
                'symbol': symbol,
                'current_price': historical_data['Close'].iloc[-1],
                'prediction': prediction,
                'sentiment': sentiment,
                'trading_signal': signal,
                'technical_indicators': technical_indicators,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error generating predictions for {symbol}: {e}")
            return None

# Initialize dashboard manager
dashboard_manager = DashboardManager()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/market-overview')
def get_market_overview():
    overview = dashboard_manager.get_market_overview()
    return jsonify(overview)

@app.route('/api/stock-data/<symbol>')
def get_stock_data(symbol):
    chart_data = dashboard_manager.generate_stock_charts(symbol)
    predictions = dashboard_manager.generate_ai_predictions(symbol)
    
    return jsonify({
        'chart_data': chart_data,
        'predictions': predictions,
        'symbol': symbol
    })

@app.route('/api/portfolio')
def get_portfolio_data():
    portfolio_data = dashboard_manager.get_portfolio_metrics()
    return jsonify(portfolio_data)

@app.route('/api/predictions/<symbol>')
def get_predictions(symbol):
    predictions = dashboard_manager.generate_ai_predictions(symbol)
    return jsonify(predictions)

@app.route('/api/trading-signal', methods=['POST'])
def generate_trading_signal():
    data = request.json
    symbol = data.get('symbol')
    
    predictions = dashboard_manager.generate_ai_predictions(symbol)
    if predictions:
        return jsonify({
            'signal': predictions['trading_signal'],
            'confidence': predictions['prediction']['confidence'],
            'recommendation': _get_trading_recommendation(predictions['trading_signal'])
        })
    
    return jsonify({'error': 'Could not generate signal'})

def _get_trading_recommendation(signal):
    """Convert signal to trading recommendation"""
    recommendations = {
        'BUY': 'Strong Buy Recommendation',
        'SELL': 'Strong Sell Recommendation',
        'WEAK_BUY': 'Consider Buying',
        'WEAK_SELL': 'Consider Selling',
        'HOLD': 'Hold Position'
    }
    return recommendations.get(signal, 'No Clear Signal')

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'status': 'Connected to real-time dashboard'})

@socketio.on('subscribe_stock')
def handle_subscribe_stock(data):
    symbol = data.get('symbol')
    print(f'Client subscribed to {symbol}')
    
    # Start sending real-time updates for this symbol
    def send_updates():
        while True:
            try:
                predictions = dashboard_manager.generate_ai_predictions(symbol)
                if predictions:
                    emit('stock_update', {
                        'symbol': symbol,
                        'data': predictions,
                        'timestamp': datetime.now().isoformat()
                    })
                time.sleep(10)  # Update every 10 seconds
            except Exception as e:
                print(f"Error in real-time updates for {symbol}: {e}")
                time.sleep(5)
    
    # Start background thread for this symbol
    thread = threading.Thread(target=send_updates)
    thread.daemon = True
    thread.start()

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# Background task for market data updates
def update_market_data():
    """Background task to update market data"""
    while True:
        try:
            # Update portfolio value
            portfolio_data = dashboard_manager.get_portfolio_metrics()
            socketio.emit('portfolio_update', portfolio_data)
            
            # Update market overview
            overview = dashboard_manager.get_market_overview()
            socketio.emit('market_overview', overview)
            
            time.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            print(f"Error in market data updates: {e}")
            time.sleep(10)

if __name__ == '__main__':
    print("🚀 Starting AI Stock Predictor Dashboard...")
    print("📊 Open: http://localhost:5000")
    
    # Start background tasks
    market_thread = threading.Thread(target=update_market_data)
    market_thread.daemon = True
    market_thread.start()
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)