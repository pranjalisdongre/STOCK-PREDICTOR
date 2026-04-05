import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    def __init__(self, max_portfolio_risk=0.10, max_position_risk=0.02, max_drawdown=0.15):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_drawdown = max_drawdown
        self.risk_history = []
        self.alerts = []
    
    def assess_market_conditions(self, market_data, volatility_lookback=20):
        """Assess overall market conditions for risk"""
        if market_data.empty:
            return 'NORMAL'
        
        # Calculate market volatility
        returns = market_data['Close'].pct_change().dropna()
        if len(returns) < volatility_lookback:
            return 'NORMAL'
        
        recent_volatility = returns.tail(volatility_lookback).std()
        historical_volatility = returns.std()
        
        volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1
        
        # Market trend
        price_trend = self._calculate_trend(market_data['Close'])
        
        # Determine market regime
        if volatility_ratio > 1.5:
            regime = 'HIGH_VOLATILITY'
        elif volatility_ratio < 0.7:
            regime = 'LOW_VOLATILITY'
        elif price_trend < -0.05:  # 5% downtrend
            regime = 'BEARISH'
        elif price_trend > 0.05:   # 5% uptrend
            regime = 'BULLISH'
        else:
            regime = 'NORMAL'
        
        return regime
    
    def _calculate_trend(self, prices, window=20):
        """Calculate price trend"""
        if len(prices) < window:
            return 0
        
        recent_prices = prices.tail(window)
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]
        trend = slope / recent_prices.mean()
        
        return trend
    
    def calculate_position_limits(self, market_regime, symbol_volatility):
        """Calculate position limits based on market conditions"""
        base_limit = self.max_position_risk
        
        if market_regime == 'HIGH_VOLATILITY':
            position_limit = base_limit * 0.5  # Reduce position size
            leverage_multiplier = 0.5
        elif market_regime == 'LOW_VOLATILITY':
            position_limit = base_limit * 1.2  # Slightly increase position size
            leverage_multiplier = 1.2
        elif market_regime == 'BEARISH':
            position_limit = base_limit * 0.7  # Conservative in bear markets
            leverage_multiplier = 0.7
        elif market_regime == 'BULLISH':
            position_limit = base_limit * 1.1  # Slightly aggressive in bull markets
            leverage_multiplier = 1.1
        else:
            position_limit = base_limit
            leverage_multiplier = 1.0
        
        # Adjust for symbol-specific volatility
        if symbol_volatility > 0.03:  # High volatility symbol
            position_limit *= 0.7
        elif symbol_volatility < 0.01:  # Low volatility symbol
            position_limit *= 1.2
        
        return {
            'position_limit': position_limit,
            'leverage_multiplier': leverage_multiplier,
            'max_position_value': None,  # Would be calculated based on portfolio
            'stop_loss_adjustment': self._calculate_stop_loss_adjustment(market_regime, symbol_volatility)
        }
    
    def _calculate_stop_loss_adjustment(self, market_regime, volatility):
        """Calculate appropriate stop loss levels"""
        base_stop_loss = 0.04  # 4% base stop loss
        
        if market_regime == 'HIGH_VOLATILITY':
            return base_stop_loss * 1.5  # Wider stops in high volatility
        elif market_regime == 'LOW_VOLATILITY':
            return base_stop_loss * 0.7   # Tighter stops in low volatility
        
        # Adjust for volatility
        volatility_multiplier = volatility / 0.02  # Normalize to 2% volatility
        return base_stop_loss * max(0.5, min(2.0, volatility_multiplier))
    
    def validate_trade(self, trade_signal, portfolio, market_data, symbol_data):
        """Validate if a trade should be executed based on risk parameters"""
        validation_result = {
            'approved': True,
            'adjusted_size': None,
            'risk_level': 'LOW',
            'reasons': []
        }
        
        # Check portfolio concentration
        if self._is_over_concentrated(portfolio, trade_signal['symbol']):
            validation_result['approved'] = False
            validation_result['reasons'].append('Portfolio over-concentrated')
        
        # Check maximum drawdown limit
        if self._exceeds_max_drawdown(portfolio):
            validation_result['approved'] = False
            validation_result['reasons'].append('Maximum drawdown exceeded')
        
        # Check market hours (avoid trading in extended hours for retail)
        if self._is_off_hours():
            validation_result['reasons'].append('Trading in off-hours')
        
        # Check volatility limits
        symbol_volatility = symbol_data['Close'].pct_change().std()
        if symbol_volatility > 0.05:  # 5% daily volatility
            validation_result['risk_level'] = 'HIGH'
            validation_result['reasons'].append('High volatility symbol')
        
        # Adjust position size if needed
        if validation_result['approved']:
            validation_result['adjusted_size'] = self._calculate_adjusted_position_size(
                trade_signal, portfolio, symbol_volatility
            )
        
        # Log risk assessment
        self._log_risk_assessment(trade_signal, validation_result)
        
        return validation_result
    
    def _is_over_concentrated(self, portfolio, symbol):
        """Check if portfolio is over-concentrated in a single symbol"""
        if not portfolio.get('positions'):
            return False
        
        total_value = portfolio.get('portfolio_value', 1)
        symbol_value = sum(
            pos['position_size'] * pos.get('current_price', pos['entry_price'])
            for sym, pos in portfolio['positions'].items() if sym == symbol
        )
        
        concentration = symbol_value / total_value
        return concentration > 0.2  # Max 20% in single symbol
    
    def _exceeds_max_drawdown(self, portfolio):
        """Check if portfolio exceeds maximum drawdown"""
        peak_value = portfolio.get('peak_value', portfolio.get('portfolio_value', 1))
        current_value = portfolio.get('portfolio_value', 1)
        
        drawdown = (peak_value - current_value) / peak_value
        return drawdown > self.max_drawdown
    
    def _is_off_hours(self):
        """Check if current time is outside regular trading hours"""
        now = datetime.now()
        # Simple check - in production, use market calendar
        if now.weekday() >= 5:  # Weekend
            return True
        
        hour = now.hour
        if hour < 9 or hour >= 16:  # Outside 9 AM - 4 PM
            return True
        
        return False
    
    def _calculate_adjusted_position_size(self, trade_signal, portfolio, symbol_volatility):
        """Calculate risk-adjusted position size"""
        base_size = trade_signal.get('position_size', 0)
        portfolio_value = portfolio.get('portfolio_value', 1)
        
        # Calculate maximum allowed position value
        max_position_value = portfolio_value * self.max_position_risk
        
        # Adjust for volatility
        volatility_factor = 0.02 / max(symbol_volatility, 0.01)  # Normalize to 2% volatility
        volatility_factor = max(0.5, min(2.0, volatility_factor))
        
        adjusted_size = base_size * volatility_factor
        
        # Ensure we don't exceed maximum position value
        position_value = adjusted_size * trade_signal.get('price', 1)
        if position_value > max_position_value:
            adjusted_size = max_position_value / trade_signal.get('price', 1)
        
        return adjusted_size
    
    def _log_risk_assessment(self, trade_signal, validation_result):
        """Log risk assessment for monitoring"""
        assessment = {
            'timestamp': datetime.now(),
            'symbol': trade_signal.get('symbol'),
            'signal': trade_signal.get('signal'),
            'approved': validation_result['approved'],
            'risk_level': validation_result['risk_level'],
            'reasons': validation_result['reasons'],
            'adjusted_size': validation_result['adjusted_size']
        }
        
        self.risk_history.append(assessment)
        
        # Generate alerts for high-risk trades
        if validation_result['risk_level'] == 'HIGH':
            alert = {
                'timestamp': datetime.now(),
                'symbol': trade_signal.get('symbol'),
                'type': 'HIGH_RISK_TRADE',
                'message': f"High risk trade detected for {trade_signal.get('symbol')}",
                'details': validation_result
            }
            self.alerts.append(alert)
    
    def get_risk_report(self, portfolio, market_data):
        """Generate comprehensive risk report"""
        report = {
            'timestamp': datetime.now(),
            'portfolio_risk': self._calculate_portfolio_risk(portfolio),
            'market_conditions': self.assess_market_conditions(market_data),
            'current_alerts': self.alerts[-10:],  # Last 10 alerts
            'risk_metrics': {
                'var_95': self._calculate_var(portfolio, confidence=0.95),
                'expected_shortfall': self._calculate_expected_shortfall(portfolio),
                'beta': self._calculate_portfolio_beta(portfolio, market_data)
            },
            'recommendations': self._generate_risk_recommendations(portfolio)
        }
        
        return report
    
    def _calculate_portfolio_risk(self, portfolio):
        """Calculate overall portfolio risk score"""
        # Simplified risk calculation
        risk_score = 0
        
        # Concentration risk
        if portfolio.get('positions'):
            positions = portfolio['positions']
            total_value = portfolio['portfolio_value']
            concentration = sum(
                pos['position_size'] * pos.get('current_price', pos['entry_price'])
                for pos in positions.values()
            ) / total_value
            
            if concentration > 0.8:
                risk_score += 3
            elif concentration > 0.5:
                risk_score += 2
            elif concentration > 0.3:
                risk_score += 1
        
        # Drawdown risk
        peak_value = portfolio.get('peak_value', portfolio['portfolio_value'])
        drawdown = (peak_value - portfolio['portfolio_value']) / peak_value
        if drawdown > 0.1:
            risk_score += 2
        elif drawdown > 0.05:
            risk_score += 1
        
        return min(risk_score, 5)  # Scale 0-5

# Test function
def test_risk_manager():
    """Test the risk manager"""
    print("🧪 Testing Risk Manager...")
    
    risk_manager = RiskManager()
    
    # Sample market data
    market_data = pd.DataFrame({
        'Close': np.random.normal(100, 5, 100).cumsum() + 1000
    })
    
    # Test market condition assessment
    market_regime = risk_manager.assess_market_conditions(market_data)
    print(f"📊 Market regime: {market_regime}")
    
    # Test position limits
    limits = risk_manager.calculate_position_limits(market_regime, 0.02)
    print(f"📈 Position limits: {limits}")
    
    # Sample portfolio
    sample_portfolio = {
        'portfolio_value': 10000,
        'cash': 5000,
        'positions': {
            'AAPL': {'position_size': 10, 'entry_price': 150, 'current_price': 155},
            'GOOGL': {'position_size': 5, 'entry_price': 2500, 'current_price': 2550}
        },
        'peak_value': 11000
    }
    
    # Sample trade signal
    trade_signal = {
        'symbol': 'TSLA',
        'signal': 'BUY',
        'position_size': 50,
        'price': 200
    }
    
    # Test trade validation
    validation = risk_manager.validate_trade(trade_signal, sample_portfolio, market_data, market_data)
    print(f"✅ Trade validation: {validation}")
    
    print("✅ Risk manager test completed!")
    return risk_manager

if __name__ == "__main__":
    test_risk_manager()