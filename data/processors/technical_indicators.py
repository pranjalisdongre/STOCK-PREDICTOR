import pandas as pd
import numpy as np
import talib
from typing import List, Dict

class TechnicalIndicatorProcessor:
    def __init__(self):
        self.indicators_config = {
            'trend': ['SMA', 'EMA', 'MACD', 'ADX'],
            'momentum': ['RSI', 'Stoch', 'WilliamsR', 'CCI'],
            'volatility': ['BBANDS', 'ATR', 'NATR'],
            'volume': ['OBV', 'AD', 'ADOSC']
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"⚠️  Missing required column: {col}")
                return df
        
        try:
            # Trend Indicators
            df = self._calculate_trend_indicators(df)
            
            # Momentum Indicators  
            df = self._calculate_momentum_indicators(df)
            
            # Volatility Indicators
            df = self._calculate_volatility_indicators(df)
            
            # Volume Indicators
            df = self._calculate_volume_indicators(df)
            
            # Custom Indicators
            df = self._calculate_custom_indicators(df)
            
            print(f"✅ Calculated {len([col for col in df.columns if col not in required_cols])} technical indicators")
            
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
        
        return df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-following indicators"""
        # Moving Averages
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
        
        df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'])
        
        # ADX (Average Directional Index)
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        # RSI
        df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
        df['RSI_21'] = talib.RSI(df['Close'], timeperiod=21)
        
        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(
            df['High'], df['Low'], df['Close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # Williams %R
        df['Williams_R'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # CCI (Commodity Channel Index)
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Rate of Change
        df['ROC_10'] = talib.ROC(df['Close'], timeperiod=10)
        df['ROC_21'] = talib.ROC(df['Close'], timeperiod=21)
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
            df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Average True Range
        df['ATR_14'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Normalized ATR
        df['NATR_14'] = talib.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        # On Balance Volume
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        
        # Accumulation/Distribution
        df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # A/D Oscillator
        df['ADOSC'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'])
        
        return df
    
    def _calculate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom technical indicators"""
        # Price relative to moving averages
        if 'SMA_20' in df.columns:
            df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
        
        if 'SMA_50' in df.columns:
            df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50'] * 100
        
        # Bollinger Band position
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # RSI-based signals
        if 'RSI_14' in df.columns:
            df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(int)
            df['RSI_Oversold'] = (df['RSI_14'] < 30).astype(int)
        
        # MACD signals
        if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
            df['MACD_Signal_Cross'] = (df['MACD'] > df['MACD_Signal']).astype(int)
            df['MACD_Histogram_Change'] = df['MACD_Hist'].diff()
        
        # Volatility regime
        if 'ATR_14' in df.columns:
            df['ATR_Ratio'] = df['ATR_14'] / df['Close'] * 100
            df['High_Volatility'] = (df['ATR_Ratio'] > df['ATR_Ratio'].rolling(20).mean()).astype(int)
        
        return df
    
    def get_trading_signals(self, df: pd.DataFrame) -> Dict:
        """Generate trading signals based on technical indicators"""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        signals = {}
        
        # Trend signals
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            sma_signal = 1 if latest['SMA_20'] > latest['SMA_50'] else -1
            signals['trend_sma'] = sma_signal
        
        # RSI signals
        if 'RSI_14' in df.columns:
            if latest['RSI_14'] > 70:
                signals['rsi'] = -1  # Overbought
            elif latest['RSI_14'] < 30:
                signals['rsi'] = 1   # Oversold
            else:
                signals['rsi'] = 0   # Neutral
        
        # MACD signals
        if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal']:
                signals['macd'] = 1
            else:
                signals['macd'] = -1
        
        # Bollinger Band signals
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
            if latest['Close'] > latest['BB_Upper']:
                signals['bollinger'] = -1  # Overbought
            elif latest['Close'] < latest['BB_Lower']:
                signals['bollinger'] = 1   # Oversold
            else:
                signals['bollinger'] = 0   # Neutral
        
        # Composite signal (weighted average)
        if signals:
            signals['composite'] = sum(signals.values()) / len(signals)
        
        return signals
    
    def get_indicator_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of all technical indicators"""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        summary = {}
        
        # Categorize indicators by type
        for col in df.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol', 'Date']:
                summary[col] = latest[col] if pd.notna(latest[col]) else 0
        
        return summary

# Test function
def test_technical_indicators():
    """Test technical indicator calculations"""
    processor = TechnicalIndicatorProcessor()
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.normal(100, 5, 100).cumsum() + 1000,
        'High': np.random.normal(102, 5, 100).cumsum() + 1000,
        'Low': np.random.normal(98, 5, 100).cumsum() + 1000,
        'Close': np.random.normal(100, 5, 100).cumsum() + 1000,
        'Volume': np.random.normal(1000000, 100000, 100)
    })
    
    print("Testing technical indicators...")
    processed_data = processor.calculate_all_indicators(sample_data)
    
    print(f"Original columns: {sample_data.columns.tolist()}")
    print(f"After processing: {len(processed_data.columns)} columns")
    print(f"New columns: {[col for col in processed_data.columns if col not in sample_data.columns]}")
    
    # Test trading signals
    signals = processor.get_trading_signals(processed_data)
    print(f"Trading signals: {signals}")
    
    return processor, processed_data

if __name__ == "__main__":
    test_technical_indicators()