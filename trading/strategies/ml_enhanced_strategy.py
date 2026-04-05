import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLEnhancedTradingStrategy:
    def __init__(self, initial_capital=10000, risk_per_trade=0.02):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.positions = {}
        self.trade_history = []
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        
    def calculate_position_size(self, current_price, stop_loss_price):
        """Calculate position size based on risk management"""
        risk_amount = self.portfolio_value * self.risk_per_trade
        price_risk = abs(current_price - stop_loss_price)
        
        if price_risk > 0:
            position_size = risk_amount / price_risk
            return min(position_size, self.cash / current_price)  # Can't exceed available cash
        return 0
    
    def generate_signals(self, data, predictions, technical_indicators):
        """Generate trading signals using ML predictions and technical analysis"""
        if len(data) < 2:
            return 'HOLD'
        
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2]
        
        # ML Prediction Signal
        ml_signal = self._get_ml_signal(predictions, current_price)
        
        # Technical Analysis Signals
        technical_signal = self._get_technical_signals(technical_indicators)
        
        # Sentiment Signal (if available)
        sentiment_signal = self._get_sentiment_signal(data)
        
        # Combine signals
        final_signal = self._combine_signals(ml_signal, technical_signal, sentiment_signal)
        
        return final_signal
    
    def _get_ml_signal(self, predictions, current_price):
        """Generate signal based on ML predictions"""
        if not predictions:
            return 'HOLD'
        
        ensemble_pred = predictions.get('ensemble_prediction', current_price)
        confidence = predictions.get('confidence', 0)
        
        # Only act on high-confidence predictions
        if confidence < 0.6:
            return 'HOLD'
        
        # Buy if prediction is significantly higher
        if ensemble_pred > current_price * 1.02:  # 2% expected gain
            return 'BUY'
        # Sell if prediction is significantly lower
        elif ensemble_pred < current_price * 0.98:  # 2% expected loss
            return 'SELL'
        
        return 'HOLD'
    
    def _get_technical_signals(self, indicators):
        """Generate signals based on technical indicators"""
        if not indicators:
            return 'HOLD'
        
        buy_signals = 0
        sell_signals = 0
        
        # RSI signals
        rsi = indicators.get('RSI_14', 50)
        if rsi < 30:
            buy_signals += 1
        elif rsi > 70:
            sell_signals += 1
        
        # MACD signals
        macd = indicators.get('MACD', 0)
        macd_signal = indicators.get('MACD_Signal', 0)
        if macd > macd_signal:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # Moving Average signals
        price_vs_sma20 = indicators.get('Price_vs_SMA20', 0)
        if price_vs_sma20 > 0:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # Bollinger Band signals
        bb_position = indicators.get('BB_Position', 0.5)
        if bb_position < 0.2:  # Near lower band
            buy_signals += 1
        elif bb_position > 0.8:  # Near upper band
            sell_signals += 1
        
        if buy_signals >= 3:
            return 'BUY'
        elif sell_signals >= 3:
            return 'SELL'
        
        return 'HOLD'
    
    def _get_sentiment_signal(self, data):
        """Generate signal based on sentiment analysis"""
        if 'sentiment_score' not in data.columns:
            return 'HOLD'
        
        sentiment = data['sentiment_score'].iloc[-1]
        
        if sentiment > 0.2:  # Strong positive sentiment
            return 'BUY'
        elif sentiment < -0.2:  # Strong negative sentiment
            return 'SELL'
        
        return 'HOLD'
    
    def _combine_signals(self, ml_signal, technical_signal, sentiment_signal):
        """Combine multiple signals into final decision"""
        signals = [ml_signal, technical_signal, sentiment_signal]
        
        # Count signals
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        
        # Strong buy if at least 2 buy signals and no sell signals
        if buy_count >= 2 and sell_count == 0:
            return 'BUY'
        # Strong sell if at least 2 sell signals and no buy signals
        elif sell_count >= 2 and buy_count == 0:
            return 'SELL'
        # Weak buy
        elif buy_count == 1 and sell_count == 0:
            return 'WEAK_BUY'
        # Weak sell
        elif sell_count == 1 and buy_count == 0:
            return 'WEAK_SELL'
        
        return 'HOLD'
    
    def execute_trade(self, symbol, signal, data, predictions):
        """Execute trade based on signal"""
        current_price = data['Close'].iloc[-1]
        timestamp = data.index[-1] if hasattr(data.index, 'dtype') else datetime.now()
        
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Check exit conditions for existing position
            if self._should_exit_position(position, current_price, predictions):
                return self._exit_position(symbol, current_price, timestamp)
        
        # Enter new position
        if signal in ['BUY', 'WEAK_BUY'] and symbol not in self.positions:
            return self._enter_position(symbol, signal, current_price, timestamp, predictions)
        
        return None
    
    def _should_exit_position(self, position, current_price, predictions):
        """Check if we should exit existing position"""
        entry_price = position['entry_price']
        current_pnl = (current_price - entry_price) / entry_price
        
        # Take profit
        if current_pnl > 0.08:  # 8% profit
            return True
        
        # Stop loss
        if current_pnl < -0.04:  # 4% loss
            return True
        
        # ML exit signal
        if predictions.get('ensemble_prediction', current_price) < current_price * 0.99:
            return True
        
        return False
    
    def _enter_position(self, symbol, signal, current_price, timestamp, predictions):
        """Enter a new position"""
        # Calculate stop loss based on volatility
        atr = predictions.get('technical_indicators', {}).get('ATR_14', current_price * 0.02)
        stop_loss = current_price - (atr * 1.5)
        
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        if position_size == 0:
            return None
        
        cost = position_size * current_price
        if cost > self.cash:
            return None
        
        # Create position
        position = {
            'symbol': symbol,
            'entry_price': current_price,
            'position_size': position_size,
            'entry_time': timestamp,
            'stop_loss': stop_loss,
            'take_profit': current_price * 1.08,  # 8% target
            'signal_strength': 'STRONG' if signal == 'BUY' else 'WEAK'
        }
        
        self.positions[symbol] = position
        self.cash -= cost
        
        # Record trade
        trade = {
            'symbol': symbol,
            'action': 'BUY',
            'price': current_price,
            'size': position_size,
            'timestamp': timestamp,
            'type': 'ENTRY',
            'signal': signal
        }
        self.trade_history.append(trade)
        
        return trade
    
    def _exit_position(self, symbol, current_price, timestamp):
        """Exit an existing position"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position_size = position['position_size']
        
        # Calculate P&L
        entry_price = position['entry_price']
        pnl = (current_price - entry_price) * position_size
        pnl_percent = (current_price - entry_price) / entry_price
        
        # Update portfolio
        self.cash += position_size * current_price
        self.portfolio_value = self.cash + sum(
            pos['position_size'] * current_price  # Simplified - would use current prices
            for sym, pos in self.positions.items() if sym != symbol
        )
        
        # Record trade
        trade = {
            'symbol': symbol,
            'action': 'SELL',
            'price': current_price,
            'size': position_size,
            'timestamp': timestamp,
            'type': 'EXIT',
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'holding_period': (timestamp - position['entry_time']).days
        }
        self.trade_history.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        return trade
    
    def get_portfolio_summary(self):
        """Get current portfolio summary"""
        total_value = self.cash
        total_invested = 0
        open_pnl = 0
        
        for symbol, position in self.positions.items():
            # In real implementation, you'd get current price from market data
            current_price = position['entry_price']  # Placeholder
            position_value = position['position_size'] * current_price
            total_value += position_value
            total_invested += position['position_size'] * position['entry_price']
            open_pnl += (current_price - position['entry_price']) * position['position_size']
        
        return {
            'portfolio_value': total_value,
            'cash': self.cash,
            'total_invested': total_invested,
            'open_pnl': open_pnl,
            'open_positions': len(self.positions),
            'total_trades': len([t for t in self.trade_history if t['type'] == 'EXIT'])
        }
    
    def get_performance_metrics(self):
        """Calculate trading performance metrics"""
        if not self.trade_history:
            return {}
        
        closed_trades = [t for t in self.trade_history if t['type'] == 'EXIT']
        
        if not closed_trades:
            return {}
        
        # Basic metrics
        total_pnl = sum(trade['pnl'] for trade in closed_trades)
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Risk-adjusted metrics
        returns = [t['pnl_percent'] for t in closed_trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        return {
            'total_pnl': total_pnl,
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }

# Test function
def test_trading_strategy():
    """Test the trading strategy"""
    print("🧪 Testing ML Enhanced Trading Strategy...")
    
    strategy = MLEnhancedTradingStrategy(initial_capital=10000)
    
    # Sample data
    sample_data = pd.DataFrame({
        'Close': [100, 102, 101, 105, 103, 108, 107, 110, 109, 112],
        'sentiment_score': [0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.35, 0.5, 0.45, 0.6]
    })
    
    # Sample predictions
    sample_predictions = {
        'ensemble_prediction': 115,
        'confidence': 0.75,
        'technical_indicators': {
            'RSI_14': 65,
            'MACD': 0.5,
            'MACD_Signal': 0.3,
            'Price_vs_SMA20': 0.02,
            'BB_Position': 0.6,
            'ATR_14': 2.5
        }
    }
    
    # Test signal generation
    signal = strategy.generate_signals(sample_data, sample_predictions, sample_predictions['technical_indicators'])
    print(f"📈 Generated signal: {signal}")
    
    # Test portfolio summary
    summary = strategy.get_portfolio_summary()
    print(f"💰 Portfolio summary: {summary}")
    
    print("✅ Trading strategy test completed!")
    return strategy

if __name__ == "__main__":
    test_trading_strategy()