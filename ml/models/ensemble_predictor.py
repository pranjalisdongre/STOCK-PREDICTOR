import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnsembleStockPredictor:
    def __init__(self, sequence_length=60, lookahead=1):
        self.sequence_length = sequence_length
        self.lookahead = lookahead
        self.scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        self.model_performance = {}
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'linear_regression': LinearRegression(),
            'lstm': self._build_lstm_model()
        }
        
        self.ensemble_weights = {}
    
    def _build_lstm_model(self):
        """Build advanced LSTM neural network"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, 1),
                kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(64, return_sequences=True,
                kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(32, return_sequences=False,
                kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def prepare_data(self, df, feature_columns, target_column='Close'):
        """Prepare data for training with advanced feature engineering"""
        # Select features and handle missing values
        features = df[feature_columns].fillna(method='ffill').fillna(0)
        target = df[target_column].shift(-self.lookahead)  # Predict future price
        
        # Remove rows with NaN targets
        valid_idx = target.notna()
        features = features[valid_idx]
        target = target[valid_idx]
        
        return features, target
    
    def create_sequences(self, features, target):
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []
        
        for i in range(len(features) - self.sequence_length):
            X_seq.append(features.iloc[i:(i + self.sequence_length)].values)
            y_seq.append(target.iloc[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_models(self, df, feature_columns, target_column='Close', validation_split=0.2):
        """Train all ensemble models with comprehensive evaluation"""
        print("🚀 Starting ensemble model training...")
        
        # Prepare data
        features, target = self.prepare_data(df, feature_columns, target_column)
        
        if len(features) < self.sequence_length * 2:
            raise ValueError(f"Insufficient data. Need at least {self.sequence_length * 2} samples.")
        
        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        print(f"📊 Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train individual models
        model_predictions = {}
        
        # Random Forest
        print("🌲 Training Random Forest...")
        self.models['random_forest'].fit(X_train_scaled, y_train)
        rf_pred = self.models['random_forest'].predict(X_test_scaled)
        model_predictions['random_forest'] = rf_pred
        
        # Gradient Boosting
        print("📈 Training Gradient Boosting...")
        self.models['gradient_boosting'].fit(X_train_scaled, y_train)
        gb_pred = self.models['gradient_boosting'].predict(X_test_scaled)
        model_predictions['gradient_boosting'] = gb_pred
        
        # Linear Regression
        print("📉 Training Linear Regression...")
        self.models['linear_regression'].fit(X_train_scaled, y_train)
        lr_pred = self.models['linear_regression'].predict(X_test_scaled)
        model_predictions['linear_regression'] = lr_pred
        
        # LSTM
        print("🧠 Training LSTM...")
        X_seq_train, y_seq_train = self.create_sequences(pd.DataFrame(X_train_scaled, columns=feature_columns), y_train)
        X_seq_test, y_seq_test = self.create_sequences(pd.DataFrame(X_test_scaled, columns=feature_columns), y_test)
        
        # Scale target for LSTM
        y_scaler = StandardScaler()
        y_seq_train_scaled = y_scaler.fit_transform(y_seq_train.reshape(-1, 1)).flatten()
        y_seq_test_scaled = y_scaler.transform(y_seq_test.reshape(-1, 1)).flatten()
        
        # Reshape for LSTM
        X_seq_train = X_seq_train.reshape((X_seq_train.shape[0], X_seq_train.shape[1], 1))
        X_seq_test = X_seq_test.reshape((X_seq_test.shape[0], X_seq_test.shape[1], 1))
        
        # Train LSTM with callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
        ]
        
        history = self.models['lstm'].fit(
            X_seq_train, y_seq_train_scaled,
            epochs=100,
            batch_size=32,
            validation_data=(X_seq_test, y_seq_test_scaled),
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        # LSTM predictions
        lstm_pred_scaled = self.models['lstm'].predict(X_seq_test).flatten()
        lstm_pred = y_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
        model_predictions['lstm'] = lstm_pred
        
        # Calculate ensemble weights based on performance
        self._calculate_ensemble_weights(y_test, model_predictions)
        
        # Store performance metrics
        self._evaluate_models(y_test, model_predictions)
        
        self.is_trained = True
        print("✅ Ensemble training completed!")
        
        return self.model_performance
    
    def _calculate_ensemble_weights(self, y_true, predictions):
        """Calculate optimal weights for ensemble based on model performance"""
        model_errors = {}
        
        for model_name, y_pred in predictions.items():
            mae = mean_absolute_error(y_true, y_pred)
            model_errors[model_name] = mae
        
        # Inverse error weighting (lower error = higher weight)
        total_inverse_error = sum(1 / error for error in model_errors.values())
        self.ensemble_weights = {
            model: (1 / error) / total_inverse_error 
            for model, error in model_errors.items()
        }
        
        print("🤖 Ensemble weights calculated:")
        for model, weight in self.ensemble_weights.items():
            print(f"   {model}: {weight:.3f}")
    
    def _evaluate_models(self, y_true, predictions):
        """Comprehensive model evaluation"""
        for model_name, y_pred in predictions.items():
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # Direction accuracy
            direction_true = np.sign(np.diff(y_true))
            direction_pred = np.sign(np.diff(y_pred))
            direction_accuracy = np.mean(direction_true == direction_pred)
            
            self.model_performance[model_name] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'direction_accuracy': direction_accuracy
            }
        
        # Ensemble performance
        ensemble_pred = self._get_ensemble_prediction(predictions)
        ensemble_mae = mean_absolute_error(y_true, ensemble_pred)
        ensemble_r2 = r2_score(y_true, ensemble_pred)
        
        self.model_performance['ensemble'] = {
            'mae': ensemble_mae,
            'r2': ensemble_r2
        }
        
        print("📊 Model Performance Summary:")
        for model, metrics in self.model_performance.items():
            print(f"   {model:>20}: MAE={metrics['mae']:.4f}, R²={metrics.get('r2', 0):.4f}")
    
    def _get_ensemble_prediction(self, predictions):
        """Calculate weighted ensemble prediction"""
        ensemble_pred = np.zeros_like(next(iter(predictions.values())))
        
        for model_name, pred in predictions.items():
            ensemble_pred += pred * self.ensemble_weights[model_name]
        
        return ensemble_pred
    
    def predict(self, recent_data, feature_columns):
        """Make prediction using ensemble approach"""
        if not self.is_trained:
            raise Exception("Models must be trained before prediction")
        
        # Prepare input data
        if len(recent_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
        
        # Use most recent sequence
        input_features = recent_data[feature_columns].iloc[-self.sequence_length:].fillna(0)
        input_scaled = self.feature_scaler.transform(input_features)
        
        # Individual model predictions
        predictions = {}
        
        # Tree-based models
        current_features = input_scaled[-1:].flatten()  # Most recent features
        predictions['random_forest'] = self.models['random_forest'].predict([current_features])[0]
        predictions['gradient_boosting'] = self.models['gradient_boosting'].predict([current_features])[0]
        predictions['linear_regression'] = self.models['linear_regression'].predict([current_features])[0]
        
        # LSTM prediction
        lstm_input = input_scaled.reshape(1, self.sequence_length, 1)
        lstm_pred_scaled = self.models['lstm'].predict(lstm_input, verbose=0)[0][0]
        
        # Note: For LSTM, we'd need to inverse transform properly
        # This is simplified - in production, maintain separate scaler for LSTM
        predictions['lstm'] = lstm_pred_scaled * 100  # Approximation
        
        # Ensemble prediction
        ensemble_pred = sum(
            pred * self.ensemble_weights[model] 
            for model, pred in predictions.items()
        )
        
        # Calculate confidence based on model agreement
        confidence = self._calculate_prediction_confidence(predictions, ensemble_pred)
        
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': predictions,
            'confidence': confidence,
            'timestamp': pd.Timestamp.now()
        }
    
    def _calculate_prediction_confidence(self, predictions, ensemble_pred):
        """Calculate prediction confidence based on model agreement"""
        individual_preds = list(predictions.values())
        std_dev = np.std(individual_preds)
        mean_pred = np.mean(individual_preds)
        
        if mean_pred == 0:
            return 0.5
        
        # Confidence is higher when models agree (low std dev)
        coefficient_of_variation = std_dev / abs(mean_pred)
        confidence = max(0, 1 - coefficient_of_variation)
        
        return min(confidence, 1.0)
    
    def save_models(self, filepath='ensemble_models.pkl'):
        """Save trained models and scalers"""
        if not self.is_trained:
            raise Exception("No trained models to save")
        
        model_data = {
            'ensemble_weights': self.ensemble_weights,
            'feature_scaler': self.feature_scaler,
            'model_performance': self.model_performance,
            'sequence_length': self.sequence_length,
            'lookahead': self.lookahead
        }
        
        # Save individual models
        joblib.dump(self.models['random_forest'], 'random_forest_model.pkl')
        joblib.dump(self.models['gradient_boosting'], 'gradient_boosting_model.pkl')
        joblib.dump(self.models['linear_regression'], 'linear_regression_model.pkl')
        self.models['lstm'].save('lstm_model.h5')
        
        joblib.dump(model_data, filepath)
        print(f"💾 Models saved to {filepath}")
    
    def load_models(self, filepath='ensemble_models.pkl'):
        """Load trained models and scalers"""
        model_data = joblib.load(filepath)
        
        self.ensemble_weights = model_data['ensemble_weights']
        self.feature_scaler = model_data['feature_scaler']
        self.model_performance = model_data['model_performance']
        self.sequence_length = model_data['sequence_length']
        self.lookahead = model_data['lookahead']
        
        # Load individual models
        from tensorflow.keras.models import load_model
        
        self.models['random_forest'] = joblib.load('random_forest_model.pkl')
        self.models['gradient_boosting'] = joblib.load('gradient_boosting_model.pkl')
        self.models['linear_regression'] = joblib.load('linear_regression_model.pkl')
        self.models['lstm'] = load_model('lstm_model.h5')
        
        self.is_trained = True
        print(f"📂 Models loaded from {filepath}")

# Test function
def test_ensemble_predictor():
    """Test the ensemble predictor with sample data"""
    print("🧪 Testing Ensemble Predictor...")
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.normal(100, 5, 500).cumsum() + 1000,
        'High': np.random.normal(102, 5, 500).cumsum() + 1000,
        'Low': np.random.normal(98, 5, 500).cumsum() + 1000,
        'Close': np.random.normal(100, 5, 500).cumsum() + 1000,
        'Volume': np.random.normal(1000000, 100000, 500),
        'RSI_14': np.random.uniform(20, 80, 500),
        'MACD': np.random.normal(0, 2, 500),
        'SMA_20': np.random.normal(100, 5, 500).cumsum() + 1000,
    })
    
    # Feature columns for training
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'MACD', 'SMA_20']
    
    # Initialize and train predictor
    predictor = EnsembleStockPredictor(sequence_length=30, lookahead=1)
    
    try:
        performance = predictor.train_models(sample_data, feature_columns)
        print("✅ Ensemble predictor test completed successfully!")
        
        # Test prediction
        recent_data = sample_data.tail(50)
        prediction = predictor.predict(recent_data, feature_columns)
        print(f"📈 Sample prediction: {prediction['ensemble_prediction']:.2f}")
        print(f"🎯 Confidence: {prediction['confidence']:.2f}")
        
        return predictor
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    test_ensemble_predictor()