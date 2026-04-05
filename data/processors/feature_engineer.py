import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = []
    
    def create_advanced_features(self, df: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
        """Create advanced features for machine learning"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Price-based features
        df = self._create_price_features(df, target_col)
        
        # Volume-based features
        df = self._create_volume_features(df)
        
        # Technical indicator features
        df = self._create_technical_features(df)
        
        # Time-based features
        df = self._create_time_features(df)
        
        # Statistical features
        df = self._create_statistical_features(df, target_col)
        
        # Target variable for prediction
        df = self._create_target_variable(df, target_col)
        
        print(f"✅ Created {len([col for col in df.columns if 'feature' in col.lower()])} advanced features")
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create price-based features"""
        # Price movements
        df['price_change'] = df[target_col].pct_change()
        df['price_change_abs'] = df[target_col].diff().abs()
        
        # High-Low relationship
        df['hl_ratio'] = df['High'] / df['Low']
        df['hl_range'] = (df['High'] - df['Low']) / df[target_col]
        
        # Open-Close relationship
        df['oc_ratio'] = df['Close'] / df['Open']
        df['oc_range'] = (df['Close'] - df['Open']) / df['Open']
        
        # Gap features
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Momentum features
        for period in [1, 3, 5, 10]:
            df[f'momentum_{period}'] = df[target_col] / df[target_col].shift(period) - 1
            df[f'volatility_{period}'] = df[target_col].pct_change().rolling(period).std()
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        if 'Volume' not in df.columns:
            return df
            
        # Volume trends
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Volume-price relationship
        df['volume_price_correlation'] = df['Volume'].rolling(10).corr(df['Close'])
        
        # Abnormal volume
        df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
        df['high_volume'] = (df['volume_zscore'] > 2).astype(int)
        
        return df
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from technical indicators"""
        # RSI-based features
        if 'RSI_14' in df.columns:
            df['rsi_trend'] = df['RSI_14'].diff()
            df['rsi_level'] = pd.cut(df['RSI_14'], 
                                   bins=[0, 30, 70, 100], 
                                   labels=['oversold', 'neutral', 'overbought'])
        
        # Moving average features
        ma_columns = [col for col in df.columns if 'SMA' in col or 'EMA' in col]
        for ma_col in ma_columns:
            df[f'{ma_col}_position'] = (df['Close'] - df[ma_col]) / df[ma_col]
            df[f'{ma_col}_trend'] = df[ma_col].diff()
        
        # Bollinger Band features
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
            df['bb_squeeze'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
            df['bb_position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # MACD features
        if 'MACD' in df.columns:
            df['macd_trend'] = df['MACD'].diff()
            df['macd_signal_ratio'] = df['MACD'] / df['MACD_Signal']
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        if 'Date' not in df.columns and df.index.dtype == 'datetime64[ns]':
            df['Date'] = df.index
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Time features
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['day_of_month'] = df['Date'].dt.day
            df['week_of_year'] = df['Date'].dt.isocalendar().week
            df['month'] = df['Date'].dt.month
            df['quarter'] = df['Date'].dt.quarter
            df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
            
            # Seasonal features
            df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
            df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create statistical features"""
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window).std()
            df[f'rolling_skew_{window}'] = df[target_col].rolling(window).skew()
            df[f'rolling_kurt_{window}'] = df[target_col].rolling(window).kurt()
            
            # Z-score
            df[f'zscore_{window}'] = (df[target_col] - df[f'rolling_mean_{window}']) / df[f'rolling_std_{window}']
        
        # Price position relative to recent range
        for window in [5, 10, 20]:
            df[f'price_position_{window}'] = (
                (df[target_col] - df[target_col].rolling(window).min()) / 
                (df[target_col].rolling(window).max() - df[target_col].rolling(window).min())
            )
        
        return df
    
    def _create_target_variable(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create target variable for prediction"""
        # Next period return
        df['target_return_1'] = df[target_col].shift(-1) / df[target_col] - 1
        df['target_return_5'] = df[target_col].shift(-5) / df[target_col] - 1
        
        # Binary classification: will price go up?
        df['target_binary_1'] = (df['target_return_1'] > 0).astype(int)
        df['target_binary_5'] = (df['target_return_5'] > 0).astype(int)
        
        # Volatility prediction
        df['target_volatility_5'] = df[target_col].pct_change().rolling(5).std().shift(-5)
        
        return df
    
    def select_best_features(self, df: pd.DataFrame, target_col: str, k: int = 20):
        """Select best features using statistical tests"""
        # Prepare data
        feature_cols = [col for col in df.columns if col not in [
            'Date', 'Symbol', target_col, 'target_return_1', 'target_return_5',
            'target_binary_1', 'target_binary_5', 'target_volatility_5'
        ] and pd.api.types.is_numeric_dtype(df[col])]
        
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        self.selected_features = [feature_cols[i] for i in self.feature_selector.get_support(indices=True)]
        
        print(f"✅ Selected top {len(self.selected_features)} features")
        return self.selected_features
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Get feature importance scores"""
        if self.feature_selector is None:
            self.select_best_features(df, target_col)
        
        feature_cols = [col for col in df.columns if col not in [
            'Date', 'Symbol', target_col, 'target_return_1', 'target_return_5',
            'target_binary_1', 'target_binary_5', 'target_volatility_5'
        ] and pd.api.types.is_numeric_dtype(df[col])]
        
        scores = self.feature_selector.scores_
        pvalues = self.feature_selector.pvalues_
        
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'score': scores,
            'pvalue': pvalues
        }).sort_values('score', ascending=False)
        
        return importance_df

# Test function
def test_feature_engineering():
    """Test feature engineering"""
    from technical_indicators import TechnicalIndicatorProcessor
    
    # Create sample data with technical indicators
    processor = TechnicalIndicatorProcessor()
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.normal(100, 5, 100).cumsum() + 1000,
        'High': np.random.normal(102, 5, 100).cumsum() + 1000,
        'Low': np.random.normal(98, 5, 100).cumsum() + 1000,
        'Close': np.random.normal(100, 5, 100).cumsum() + 1000,
        'Volume': np.random.normal(1000000, 100000, 100)
    })
    
    # Add technical indicators
    sample_data = processor.calculate_all_indicators(sample_data)
    
    # Feature engineering
    engineer = FeatureEngineer()
    featured_data = engineer.create_advanced_features(sample_data)
    
    print(f"Original shape: {sample_data.shape}")
    print(f"After feature engineering: {featured_data.shape}")
    print(f"Target variables created: {[col for col in featured_data.columns if 'target' in col]}")
    
    # Feature selection
    selected_features = engineer.select_best_features(featured_data, 'Close', k=15)
    print(f"Selected features: {selected_features}")
    
    return engineer, featured_data

if __name__ == "__main__":
    test_feature_engineering()