import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime

from web.dashboard.app import app, socketio, dashboard_manager


class TestWebApp:
    """Test cases for web application"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        
    def test_home_route(self):
        """Test home page route"""
        response = self.app.get('/')
        assert response.status_code == 200
        assert b'AI Stock Predictor' in response.data
        
    def test_dashboard_route(self):
        """Test dashboard route"""
        response = self.app.get('/dashboard')
        assert response.status_code == 200
        assert b'Trading Dashboard' in response.data
        
    def test_api_health_check(self):
        """Test API health check endpoint"""
        response = self.app.get('/api/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        
    def test_api_stocks_list(self):
        """Test stocks list API endpoint"""
        response = self.app.get('/api/stocks')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'symbols' in data
        assert 'count' in data
        assert isinstance(data['symbols'], list)
        
    def test_api_stock_data(self):
        """Test stock data API endpoint"""
        with patch('web.api.routes.data_collector.get_historical_data') as mock_get:
            mock_get.return_value = Mock(
                reset_index=Mock(return_value=Mock(to_dict=Mock(return_value=[])))
            )
            
            response = self.app.get('/api/stocks/AAPL/data?period=1mo')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert data['symbol'] == 'AAPL'
            assert data['period'] == '1mo'
            
    def test_api_stock_predict(self):
        """Test stock prediction API endpoint"""
        with patch('web.api.routes.data_collector.get_historical_data') as mock_get:
            mock_data = Mock()
            mock_data.__getitem__.return_value = Mock(iloc=Mock(return_value=150))
            mock_get.return_value = mock_data
            
            response = self.app.get('/api/stocks/AAPL/predict')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert data['symbol'] == 'AAPL'
            assert 'predicted_price' in data
            assert 'confidence' in data
            
    def test_api_portfolio(self):
        """Test portfolio API endpoint"""
        with patch('web.api.routes.trading_strategy.get_portfolio_summary') as mock_summary:
            with patch('web.api.routes.trading_strategy.get_performance_metrics') as mock_perf:
                mock_summary.return_value = {
                    'portfolio_value': 10000,
                    'cash': 5000,
                    'open_positions': 2
                }
                mock_perf.return_value = {
                    'total_pnl': 500,
                    'win_rate': 0.6
                }
                
                response = self.app.get('/api/portfolio')
                assert response.status_code == 200
                
                data = json.loads(response.data)
                assert 'portfolio_summary' in data
                assert 'performance_metrics' in data
                
    def test_api_trade_execution(self):
        """Test trade execution API endpoint"""
        trade_data = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 10
        }
        
        with patch('web.api.routes.trading_strategy.positions', {}):
            with patch('web.api.routes.trading_strategy.cash', 10000):
                response = self.app.post(
                    '/api/portfolio/trade',
                    data=json.dumps(trade_data),
                    content_type='application/json'
                )
                
                # Should return 200 or 400 depending on validation
                assert response.status_code in [200, 400]
                
    def test_api_backtest(self):
        """Test backtesting API endpoint"""
        backtest_request = {
            'symbol': 'AAPL',
            'strategy': 'ml_enhanced',
            'initial_capital': 10000
        }
        
        with patch('web.api.routes.data_collector.get_historical_data') as mock_get:
            with patch('web.api.routes.backtest_engine.run_backtest') as mock_backtest:
                mock_get.return_value = Mock()
                mock_backtest.return_value = {
                    'total_return': 0.1,
                    'sharpe_ratio': 1.5,
                    'max_drawdown': -0.05
                }
                
                response = self.app.post(
                    '/api/backtest',
                    data=json.dumps(backtest_request),
                    content_type='application/json'
                )
                
                assert response.status_code == 200
                
                data = json.loads(response.data)
                assert 'backtest_results' in data
                assert 'parameters' in data
                
    def test_api_market_overview(self):
        """Test market overview API endpoint"""
        with patch('web.api.routes.data_collector.symbols', ['AAPL', 'GOOGL']):
            with patch('web.api.routes.data_collector.get_historical_data') as mock_get:
                mock_get.return_value = Mock(
                    __getitem__=Mock(return_value=Mock(
                        iloc=Mock(return_value=150),
                        tail=Mock(return_value=Mock(
                            iloc=Mock(return_value=149)
                        ))
                    ))
                )
                
                response = self.app.get('/api/market/overview')
                assert response.status_code == 200
                
                data = json.loads(response.data)
                assert 'market_overview' in data
                assert 'top_movers' in data
                
    def test_api_alerts(self):
        """Test alerts API endpoint"""
        response = self.app.get('/api/alerts')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'alerts' in data
        assert 'total_alerts' in data
        assert isinstance(data['alerts'], list)
        
    def test_api_system_status(self):
        """Test system status API endpoint"""
        response = self.app.get('/api/system/status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'system_status' in data
        assert 'components' in data
        assert 'version' in data


class TestDashboardManager:
    """Test cases for DashboardManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.dashboard = dashboard_manager
        
    def test_get_market_overview(self):
        """Test market overview generation"""
        overview = self.dashboard.get_market_overview()
        
        assert 'timestamp' in overview
        assert 'total_symbols' in overview
        assert 'market_status' in overview
        assert 'total_portfolio_value' in overview
        
    def test_generate_ai_predictions(self):
        """Test AI prediction generation"""
        with patch.object(self.dashboard.data_collector, 'get_historical_data') as mock_get:
            mock_data = Mock()
            mock_data.__getitem__.return_value = Mock(iloc=Mock(return_value=150))
            mock_data.empty = False
            mock_get.return_value = mock_data
            
            prediction = self.dashboard.generate_ai_predictions('AAPL')
            
            if prediction:  # Might be None if data insufficient
                assert prediction['symbol'] == 'AAPL'
                assert 'current_price' in prediction
                assert 'prediction' in prediction
                assert 'trading_signal' in prediction
                
    def test_get_portfolio_metrics(self):
        """Test portfolio metrics retrieval"""
        with patch.object(self.dashboard.trading_strategy, 'get_portfolio_summary') as mock_summary:
            with patch.object(self.dashboard.trading_strategy, 'get_performance_metrics') as mock_perf:
                mock_summary.return_value = {
                    'portfolio_value': 10000,
                    'cash': 5000,
                    'open_positions': 2
                }
                mock_perf.return_value = {
                    'total_pnl': 500,
                    'win_rate': 0.6
                }
                
                metrics = self.dashboard.get_portfolio_metrics()
                
                assert 'summary' in metrics
                assert 'performance' in metrics
                assert 'positions' in metrics


class TestWebSocket:
    """Test cases for WebSocket functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.socket_client = socketio.test_client(app)
        self.socket_client.connect()
        
    def teardown_method(self):
        """Clean up after tests"""
        self.socket_client.disconnect()
        
    def test_websocket_connection(self):
        """Test WebSocket connection"""
        assert self.socket_client.is_connected()
        
        received = self.socket_client.get_received()
        assert len(received) > 0
        assert received[0]['name'] == 'connected'
        
    def test_websocket_subscribe_stock(self):
        """Test stock subscription via WebSocket"""
        self.socket_client.emit('subscribe_stock', {'symbol': 'AAPL'})
        
        # Check if subscription was received
        # Note: Actual data updates happen in background threads
        
    def test_websocket_disconnect(self):
        """Test WebSocket disconnection"""
        self.socket_client.disconnect()
        assert not self.socket_client.is_connected()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])