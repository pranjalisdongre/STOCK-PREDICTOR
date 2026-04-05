import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tensorflow as tf

from ml.models.ensemble_predictor import EnsembleStockPredictor
from ml.models.sentiment_analyzer import AdvancedSentimentAnalyzer
from ml.training.trainer import ModelTrainer
from ml.training.hyperparameter_tuner import HyperparameterTuner


class TestEnsembleStockPredictor:
    """Test cases for EnsembleStockPredictor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.predictor = EnsembleStockPredictor(sequence_length=5, lookahead=1)
        
    def test_initialization(self):
        """Test predictor initialization"""
        assert self.predictor is not None
        assert self.predictor.sequence_length == 5
        assert self.predictor.lookahead == 1
        assert not self.predictor.is_trained
        
    def test_build_lstm_model(self):
        """Test LSTM model construction"""
        model = self.predictor._build_lstm_model()
        
        assert model is not None
        assert isinstance(model, tf.keras.Model)
        # Check model has expected layers
        assert len(model.layers) > 0
        
    def test_prepare_data(self):
        """Test data preparation"""
        sample_data = pd.DataFrame({
            'feature_1': range(10),
            'feature_2': range(10, 20),
            'Close': range(100, 110)
        })
        
        feature_columns = ['feature_1', 'feature_2']
        features, target = self.predictor.prepare_data(
            sample_data, feature_columns, 'Close'
        )
        
        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert len(features) == len(target)
        
    def test_create_sequences(self):
        """Test sequence creation for LSTM"""
        features = pd.DataFrame({
            'feature_1': range(10),
            'feature_2': range(10, 20)
        })
        target = pd.Series(range(100, 110))
        
        X_seq, y_seq = self.predictor.create_sequences(features, target)
        
        assert isinstance(X_seq, np.ndarray)
        assert isinstance(y_seq, np.ndarray)
        assert len(X_seq) == len(y_seq)
        
    def test_train_models(self):
        """Test model training (simplified)"""
        # Create sample training data
        sample_data = pd.DataFrame({
            'Open': np.random.randn(50),
            'High': np.random.randn(50),
            'Low': np.random.randn(50),
            'Close': np.random.randn(50).cumsum() + 100,
            'Volume': np.random.randint(1000000, 5000000, 50)
        })
        
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Mock the actual training to avoid long execution
        with patch.object(self.predictor.models['random_forest'], 'fit'):
            with patch.object(self.predictor.models['gradient_boosting'], 'fit'):
                with patch.object(self.predictor.models['linear_regression'], 'fit'):
                    with patch.object(self.predictor.models['lstm'], 'fit'):
                        performance = self.predictor.train_models(
                            sample_data, feature_columns, validation_split=0.3
                        )
        
        assert isinstance(performance, dict)
        
    def test_predict_without_training(self):
        """Test prediction without training should raise error"""
        sample_data = pd.DataFrame({
            'feature_1': [1, 2, 3],
            'feature_2': [4, 5, 6]
        })
        
        with pytest.raises(Exception, match="Models must be trained before prediction"):
            self.predictor.predict(sample_data, ['feature_1', 'feature_2'])


class TestAdvancedSentimentAnalyzer:
    """Test cases for AdvancedSentimentAnalyzer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = AdvancedSentimentAnalyzer()
        
    @patch('ml.models.sentiment_analyzer.pipeline')
    def test_load_pretrained_model(self, mock_pipeline):
        """Test loading pre-trained model"""
        mock_pipeline.return_value = Mock()
        
        self.analyzer.load_pretrained_model()
        
        assert self.analyzer.sentiment_pipeline is not None
        
    def test_analyze_sentiment_advanced(self):
        """Test advanced sentiment analysis"""
        test_text = "Apple reported excellent earnings with strong growth prospects."
        
        # Mock the transformer pipeline
        with patch.object(self.analyzer, 'sentiment_pipeline') as mock_pipe:
            mock_pipe.return_value = [{'label': 'positive', 'score': 0.95}]
            
            result = self.analyzer.analyze_sentiment_advanced(test_text)
            
            assert isinstance(result, dict)
            assert 'sentiment' in result
            assert 'confidence' in result
            assert 'methods_used' in result
            
    def test_analyze_insufficient_text(self):
        """Test sentiment analysis with insufficient text"""
        result = self.analyzer.analyze_sentiment_advanced("Hi")
        
        assert result['sentiment'] == 0
        assert result['confidence'] == 0


class TestModelTrainer:
    """Test cases for ModelTrainer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.trainer = ModelTrainer()
        
    def test_prepare_training_data(self):
        """Test training data preparation"""
        # Create sample historical data
        historical_data = {
            'AAPL': pd.DataFrame({
                'Open': np.random.randn(20),
                'High': np.random.randn(20),
                'Low': np.random.randn(20),
                'Close': np.random.randn(20).cumsum() + 100,
                'Volume': np.random.randint(1000000, 5000000, 20)
            })
        }
        
        processed_data = self.trainer.prepare_training_data(historical_data)
        
        assert isinstance(processed_data, dict)
        assert 'AAPL' in processed_data
        
    def test_select_features_for_training(self):
        """Test feature selection for training"""
        # Create sample processed data
        processed_data = {
            'AAPL': pd.DataFrame({
                'Close': range(100),
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                'feature_3': np.random.randn(100),
                'target_return_1': np.random.randn(100)
            })
        }
        
        features = self.trainer.select_features_for_training(
            processed_data, 'AAPL'
        )
        
        assert isinstance(features, list)
        assert all(isinstance(feat, str) for feat in features)


class TestHyperparameterTuner:
    """Test cases for HyperparameterTuner"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tuner = HyperparameterTuner()
        
    def test_tune_random_forest(self):
        """Test Random Forest hyperparameter tuning"""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        with patch('ml.training.hyperparameter_tuner.RandomizedSearchCV') as mock_search:
            mock_search.return_value.best_estimator_ = Mock()
            mock_search.return_value.best_score_ = -0.1
            mock_search.return_value.best_params_ = {'n_estimators': 100}
            
            best_model = self.tuner.tune_random_forest(X, y, cv=2, n_iter=2)
            
            assert best_model is not None
            assert 'random_forest' in self.tuner.best_params
            
    def test_optimize_ensemble_parameters(self):
        """Test ensemble parameter optimization"""
        # Create sample processed data
        processed_data = {
            'AAPL': pd.DataFrame({
                'Close': range(100),
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100)
            })
        }
        
        with patch.object(self.tuner, 'tune_random_forest'):
            with patch.object(self.tuner, 'tune_gradient_boosting'):
                best_params = self.tuner.optimize_ensemble_parameters(
                    processed_data, 'AAPL'
                )
                
                assert isinstance(best_params, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])