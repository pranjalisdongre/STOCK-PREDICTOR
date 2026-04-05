import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ml.models.ensemble_predictor import EnsembleStockPredictor
from ml.models.sentiment_analyzer import AdvancedSentimentAnalyzer
from data.processors.technical_indicators import TechnicalIndicatorProcessor
from data.processors.feature_engineer import FeatureEngineer

class ModelTrainer:
    def __init__(self):
        self.technical_processor = TechnicalIndicatorProcessor()
        self.feature_engineer = FeatureEngineer()
        self.ensemble_predictor = None
        self.sentiment_analyzer = None
        self.training_history = {}
    
    def prepare_training_data(self, historical_data, news_data=None):
        """Prepare comprehensive training data"""
        print("📊 Preparing training data...")
        
        processed_data = {}
        
        for symbol, data in historical_data.items():
            try:
                # Technical indicators
                data_with_indicators = self.technical_processor.calculate_all_indicators(data)
                
                # Feature engineering
                data_with_features = self.feature_engineer.create_advanced_features(data_with_indicators)
                
                # Add sentiment data if available
                if news_data and symbol in news_data:
                    sentiment_df = news_data[symbol]
                    if not sentiment_df.empty:
                        # Merge sentiment data (simplified - would need date alignment)
                        data_with_features = self._merge_sentiment_data(data_with_features, sentiment_df)
                
                processed_data[symbol] = data_with_features
                print(f"✅ Processed {symbol}: {len(data_with_features)} records")
                
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                continue
        
        return processed_data
    
    def _merge_sentiment_data(self, price_data, sentiment_df):
        """Merge sentiment data with price data"""
        # Simplified implementation - in production, align by dates
        try:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['published_at']).dt.date
            price_data['date'] = pd.to_datetime(price_data['Date']).dt.date
            
            # Aggregate daily sentiment
            daily_sentiment = sentiment_df.groupby('date').agg({
                'custom_sentiment': 'mean',
                'vader_compound': 'mean'
            }).reset_index()
            
            # Merge
            merged_data = pd.merge(price_data, daily_sentiment, on='date', how='left')
            merged_data['sentiment_score'] = merged_data['custom_sentiment'].fillna(0)
            merged_data['vader_sentiment'] = merged_data['vader_compound'].fillna(0)
            
            return merged_data.drop(columns=['date'], errors='ignore')
            
        except Exception as e:
            print(f"Sentiment merge error: {e}")
            return price_data
    
    def select_features_for_training(self, processed_data, target_symbol='AAPL'):
        """Select best features for model training"""
        if target_symbol not in processed_data:
            raise ValueError(f"Target symbol {target_symbol} not in processed data")
        
        target_data = processed_data[target_symbol]
        
        # Get numeric columns excluding targets and identifiers
        exclude_cols = ['Date', 'Symbol', 'target_return_1', 'target_return_5', 
                       'target_binary_1', 'target_binary_5', 'target_volatility_5']
        
        feature_columns = [col for col in target_data.columns 
                          if col not in exclude_cols 
                          and pd.api.types.is_numeric_dtype(target_data[col])]
        
        # Feature selection
        selected_features = self.feature_engineer.select_best_features(
            target_data, 'Close', k=min(20, len(feature_columns))
        )
        
        print(f"🎯 Selected {len(selected_features)} features for training:")
        for feature in selected_features:
            print(f"   - {feature}")
        
        return selected_features
    
    def train_ensemble_model(self, processed_data, symbol='AAPL', sequence_length=60, lookahead=1):
        """Train the ensemble model for a specific symbol"""
        print(f"🚀 Training ensemble model for {symbol}...")
        
        if symbol not in processed_data:
            raise ValueError(f"Symbol {symbol} not in processed data")
        
        data = processed_data[symbol]
        
        # Select features
        feature_columns = self.select_features_for_training(processed_data, symbol)
        
        # Initialize and train ensemble predictor
        self.ensemble_predictor = EnsembleStockPredictor(
            sequence_length=sequence_length,
            lookahead=lookahead
        )
        
        # Train models
        performance = self.ensemble_predictor.train_models(data, feature_columns)
        
        # Store training history
        self.training_history[symbol] = {
            'timestamp': datetime.now(),
            'performance': performance,
            'feature_columns': feature_columns,
            'data_points': len(data),
            'sequence_length': sequence_length,
            'lookahead': lookahead
        }
        
        print(f"✅ Ensemble model training completed for {symbol}")
        return performance
    
    def train_sentiment_analyzer(self, news_data):
        """Train the sentiment analysis model"""
        print("🤖 Training sentiment analyzer...")
        
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.sentiment_analyzer.load_pretrained_model()
        
        # Combine all news data for training
        all_news = pd.concat(news_data.values(), ignore_index=True) if news_data else pd.DataFrame()
        
        if not all_news.empty:
            self.sentiment_analyzer.train_models(all_news)
            print("✅ Sentiment analyzer training completed")
        else:
            print("⚠️  No news data available for sentiment training")
        
        return self.sentiment_analyzer
    
    def cross_validate_models(self, processed_data, symbols=None, folds=3):
        """Perform cross-validation across multiple symbols"""
        if symbols is None:
            symbols = list(processed_data.keys())[:3]  # Limit for testing
        
        cv_results = {}
        
        for symbol in symbols:
            print(f"🔍 Cross-validating {symbol}...")
            
            symbol_data = processed_data[symbol]
            fold_size = len(symbol_data) // folds
            fold_scores = []
            
            for fold in range(folds):
                # Create train-test split for this fold
                test_start = fold * fold_size
                test_end = (fold + 1) * fold_size if fold < folds - 1 else len(symbol_data)
                
                train_data = pd.concat([
                    symbol_data.iloc[:test_start],
                    symbol_data.iloc[test_end:]
                ])
                test_data = symbol_data.iloc[test_start:test_end]
                
                # Feature selection on training data only
                feature_columns = self.feature_engineer.select_best_features(
                    train_data, 'Close', k=15
                )
                
                # Train model
                predictor = EnsembleStockPredictor(sequence_length=30, lookahead=1)
                performance = predictor.train_models(train_data, feature_columns)
                
                fold_scores.append(performance.get('ensemble', {}).get('mae', 0))
            
            cv_results[symbol] = {
                'mean_mae': np.mean(fold_scores),
                'std_mae': np.std(fold_scores),
                'fold_scores': fold_scores
            }
            
            print(f"   {symbol}: MAE = {cv_results[symbol]['mean_mae']:.4f} ± {cv_results[symbol]['std_mae']:.4f}")
        
        return cv_results
    
    def save_trained_models(self, model_dir='trained_models'):
        """Save all trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        if self.ensemble_predictor and self.ensemble_predictor.is_trained:
            self.ensemble_predictor.save_models(f'{model_dir}/ensemble_predictor.pkl')
            print("💾 Ensemble predictor saved")
        
        if self.sentiment_analyzer and self.sentiment_analyzer.is_trained:
            joblib.dump(self.sentiment_analyzer, f'{model_dir}/sentiment_analyzer.pkl')
            print("💾 Sentiment analyzer saved")
        
        # Save training history
        joblib.dump(self.training_history, f'{model_dir}/training_history.pkl')
        print("💾 Training history saved")
    
    def load_trained_models(self, model_dir='trained_models'):
        """Load trained models"""
        try:
            self.ensemble_predictor = EnsembleStockPredictor()
            self.ensemble_predictor.load_models(f'{model_dir}/ensemble_predictor.pkl')
            print("📂 Ensemble predictor loaded")
        except Exception as e:
            print(f"❌ Error loading ensemble predictor: {e}")
        
        try:
            self.sentiment_analyzer = joblib.load(f'{model_dir}/sentiment_analyzer.pkl')
            print("📂 Sentiment analyzer loaded")
        except Exception as e:
            print(f"❌ Error loading sentiment analyzer: {e}")
        
        try:
            self.training_history = joblib.load(f'{model_dir}/training_history.pkl')
            print("📂 Training history loaded")
        except Exception as e:
            print(f"❌ Error loading training history: {e}")

# Test function
def test_model_trainer():
    """Test the model trainer with sample data"""
    print("🧪 Testing Model Trainer...")
    
    trainer = ModelTrainer()
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    
    sample_historical = {
        'AAPL': pd.DataFrame({
            'Date': dates,
            'Open': np.random.normal(150, 5, 200).cumsum() + 1500,
            'High': np.random.normal(152, 5, 200).cumsum() + 1500,
            'Low': np.random.normal(148, 5, 200).cumsum() + 1500,
            'Close': np.random.normal(150, 5, 200).cumsum() + 1500,
            'Volume': np.random.normal(2000000, 200000, 200)
        }),
        'GOOGL': pd.DataFrame({
            'Date': dates,
            'Open': np.random.normal(2500, 50, 200).cumsum() + 25000,
            'High': np.random.normal(2520, 50, 200).cumsum() + 25000,
            'Low': np.random.normal(2480, 50, 200).cumsum() + 25000,
            'Close': np.random.normal(2500, 50, 200).cumsum() + 25000,
            'Volume': np.random.normal(500000, 50000, 200)
        })
    }
    
    # Prepare training data
    processed_data = trainer.prepare_training_data(sample_historical)
    
    # Train ensemble model
    performance = trainer.train_ensemble_model(processed_data, 'AAPL', sequence_length=30, lookahead=1)
    
    print("✅ Model trainer test completed!")
    return trainer

if __name__ == "__main__":
    test_model_trainer()