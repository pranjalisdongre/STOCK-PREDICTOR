import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    def __init__(self):
        self.best_params = {}
        self.tuning_history = {}
    
    def tune_random_forest(self, X, y, cv=3, n_iter=20):
        """Tune Random Forest hyperparameters"""
        print("🎯 Tuning Random Forest hyperparameters...")
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        
        random_search = RandomizedSearchCV(
            rf, param_distributions=param_dist,
            n_iter=n_iter, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, verbose=1
        )
        
        random_search.fit(X, y)
        
        self.best_params['random_forest'] = random_search.best_params_
        self.tuning_history['random_forest'] = {
            'best_score': -random_search.best_score_,  # Convert back to positive MAE
            'best_params': random_search.best_params_
        }
        
        print(f"✅ Random Forest best MAE: {-random_search.best_score_:.4f}")
        return random_search.best_estimator_
    
    def tune_gradient_boosting(self, X, y, cv=3, n_iter=15):
        """Tune Gradient Boosting hyperparameters"""
        print("🎯 Tuning Gradient Boosting hyperparameters...")
        
        gb = GradientBoostingRegressor(random_state=42)
        
        param_dist = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        
        random_search = RandomizedSearchCV(
            gb, param_distributions=param_dist,
            n_iter=n_iter, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, verbose=1
        )
        
        random_search.fit(X, y)
        
        self.best_params['gradient_boosting'] = random_search.best_params_
        self.tuning_history['gradient_boosting'] = {
            'best_score': -random_search.best_score_,
            'best_params': random_search.best_params_
        }
        
        print(f"✅ Gradient Boosting best MAE: {-random_search.best_score_:.4f}")
        return random_search.best_estimator_
    
    def optimize_ensemble_parameters(self, processed_data, symbol='AAPL', feature_columns=None):
        """Optimize parameters for the entire ensemble pipeline"""
        print(f"🔧 Optimizing ensemble parameters for {symbol}...")
        
        if symbol not in processed_data:
            raise ValueError(f"Symbol {symbol} not in processed data")
        
        data = processed_data[symbol]
        
        if feature_columns is None:
            # Select features automatically
            from ml.training.trainer import ModelTrainer
            trainer = ModelTrainer()
            feature_columns = trainer.select_features_for_training(processed_data, symbol)
        
        # Prepare features and target
        X = data[feature_columns].fillna(0)
        y = data['Close'].shift(-1).fillna(method='ffill')  # Next day's close
        
        # Remove rows with NaN
        valid_idx = y.notna() & X.notna().all(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 100:
            print("⚠️  Insufficient data for hyperparameter tuning")
            return self.best_params
        
        # Tune individual models
        best_rf = self.tune_random_forest(X, y)
        best_gb = self.tune_gradient_boosting(X, y)
        
        # Store optimized models
        self.optimized_models = {
            'random_forest': best_rf,
            'gradient_boosting': best_gb
        }
        
        print("✅ Ensemble parameter optimization completed!")
        return self.best_params
    
    def get_optimized_models(self):
        """Get the optimized model instances"""
        return self.optimized_models
    
    def save_tuning_results(self, filepath='hyperparameter_tuning.pkl'):
        """Save tuning results"""
        tuning_data = {
            'best_params': self.best_params,
            'tuning_history': self.tuning_history,
            'optimized_models': self.optimized_models if hasattr(self, 'optimized_models') else {}
        }
        
        joblib.dump(tuning_data, filepath)
        print(f"💾 Tuning results saved to {filepath}")
    
    def load_tuning_results(self, filepath='hyperparameter_tuning.pkl'):
        """Load tuning results"""
        tuning_data = joblib.load(filepath)
        
        self.best_params = tuning_data['best_params']
        self.tuning_history = tuning_data['tuning_history']
        self.optimized_models = tuning_data.get('optimized_models', {})
        
        print(f"📂 Tuning results loaded from {filepath}")

# Test function
def test_hyperparameter_tuner():
    """Test the hyperparameter tuner"""
    print("🧪 Testing Hyperparameter Tuner...")
    
    tuner = HyperparameterTuner()
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(300, 10)
    y = X.dot(np.random.randn(10)) + np.random.randn(300) * 0.1
    
    # Test tuning with smaller dataset for speed
    X_sample = X[:100]
    y_sample = y[:100]
    
    # Tune models
    best_rf = tuner.tune_random_forest(X_sample, y_sample, cv=2, n_iter=5)
    best_gb = tuner.tune_gradient_boosting(X_sample, y_sample, cv=2, n_iter=5)
    
    print("✅ Hyperparameter tuner test completed!")
    return tuner

if __name__ == "__main__":
    test_hyperparameter_tuner()