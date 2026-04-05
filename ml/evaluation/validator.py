import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelValidator:
    def __init__(self):
        self.validation_results = {}
        self.plots = {}
    
    def comprehensive_validation(self, y_true, y_pred, model_name='model'):
        """Perform comprehensive model validation"""
        results = {}
        
        # Regression metrics
        results['mae'] = mean_absolute_error(y_true, y_pred)
        results['mse'] = mean_squared_error(y_true, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['r2'] = r2_score(y_true, y_pred)
        
        # Percentage errors
        percentage_errors = np.abs((y_true - y_pred) / y_true) * 100
        results['mape'] = np.mean(percentage_errors)
        results['mdape'] = np.median(percentage_errors)
        
        # Direction accuracy
        direction_true = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(y_pred))
        results['direction_accuracy'] = np.mean(direction_true == direction_pred)
        
        # Profit/loss simulation
        results['profit_simulation'] = self._simulate_profit(y_true, y_pred)
        
        self.validation_results[model_name] = results
        return results
    
    def _simulate_profit(self, y_true, y_pred):
        """Simulate trading profit based on predictions"""
        if len(y_true) < 2:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        # Simple strategy: buy when predicted to go up, sell when predicted to go down
        positions = []
        cash = 10000  # Starting capital
        shares = 0
        
        for i in range(1, len(y_pred)):
            current_price = y_true[i-1]
            next_price = y_true[i]
            prediction = y_pred[i]
            
            # Buy signal: predicted price increase > 1%
            if prediction > current_price * 1.01 and cash > 0:
                shares_to_buy = cash // current_price
                if shares_to_buy > 0:
                    shares += shares_to_buy
                    cash -= shares_to_buy * current_price
            
            # Sell signal: predicted price decrease > 1%
            elif prediction < current_price * 0.99 and shares > 0:
                cash += shares * current_price
                shares = 0
        
        # Final portfolio value
        final_value = cash + shares * y_true[-1]
        total_return = (final_value - 10000) / 10000
        
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(y_true) / y_true[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': final_value
        }
    
    def compare_models(self, models_predictions, y_true):
        """Compare multiple models"""
        comparison = {}
        
        for model_name, y_pred in models_predictions.items():
            comparison[model_name] = self.comprehensive_validation(y_true, y_pred, model_name)
        
        # Create comparison DataFrame
        metrics_df = pd.DataFrame({
            model: results for model, results in comparison.items()
        }).T
        
        return metrics_df
    
    def create_validation_plots(self, y_true, y_pred, dates=None, model_name='model'):
        """Create comprehensive validation plots"""
        if dates is None:
            dates = range(len(y_true))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Actual vs Predicted', 'Residuals', 
                          'Error Distribution', 'Cumulative Returns'),
            specs=[[{"secondary_y": False}, {}],
                   [{}, {}]]
        )
        
        # Plot 1: Actual vs Predicted
        fig.add_trace(
            go.Scatter(x=dates, y=y_true, name='Actual', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=y_pred, name='Predicted', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Plot 2: Residuals
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(x=dates, y=residuals, name='Residuals', line=dict(color='green')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=dates, y=[0]*len(dates), name='Zero', line=dict(color='black', dash='dot')),
            row=1, col=2
        )
        
        # Plot 3: Error Distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Error Distribution', nbinsx=50),
            row=2, col=1
        )
        
        # Plot 4: Cumulative Returns (simulated)
        returns = np.diff(y_true) / y_true[:-1]
        cumulative_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(x=dates[1:], y=cumulative_returns, name='Cumulative Returns', 
                      line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text=f"Model Validation - {model_name}")
        self.plots[model_name] = fig
        
        return fig
    
    def walk_forward_validation(self, data, feature_columns, target_column, 
                              initial_train_size=100, step_size=10):
        """Perform walk-forward validation"""
        predictions = []
        actuals = []
        
        for i in range(initial_train_size, len(data), step_size):
            # Split data
            train_data = data.iloc[:i]
            test_data = data.iloc[i:i+step_size]
            
            if len(test_data) == 0:
                break
            
            # Train model (simplified - would use actual model training)
            # For now, we'll use a simple moving average as placeholder
            last_train_value = train_data[target_column].iloc[-1]
            pred = [last_train_value] * len(test_data)
            
            predictions.extend(pred)
            actuals.extend(test_data[target_column].values)
        
        # Calculate metrics
        if len(predictions) > 0:
            results = self.comprehensive_validation(np.array(actuals), np.array(predictions), 'walk_forward')
            return results, predictions, actuals
        
        return {}, [], []
    
    def generate_validation_report(self, models_predictions, y_true, save_path=None):
        """Generate comprehensive validation report"""
        report = {
            'timestamp': pd.Timestamp.now(),
            'models_tested': list(models_predictions.keys()),
            'validation_results': self.validation_results,
            'summary_metrics': {}
        }
        
        # Calculate summary metrics
        for model_name, results in self.validation_results.items():
            report['summary_metrics'][model_name] = {
                'mae': results['mae'],
                'r2': results['r2'],
                'direction_accuracy': results['direction_accuracy'],
                'total_return': results['profit_simulation']['total_return']
            }
        
        # Find best model
        if report['summary_metrics']:
            best_model = min(report['summary_metrics'].items(), 
                           key=lambda x: x[1]['mae'])[0]
            report['best_model'] = best_model
        
        # Save report if path provided
        if save_path:
            import joblib
            joblib.dump(report, save_path)
            print(f"💾 Validation report saved to {save_path}")
        
        return report

# Test function
def test_model_validator():
    """Test the model validator"""
    print("🧪 Testing Model Validator...")
    
    validator = ModelValidator()
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.cumsum(np.random.randn(100)) + 100
    y_pred = y_true + np.random.randn(100) * 2  # Add some noise
    
    # Test comprehensive validation
    results = validator.comprehensive_validation(y_true, y_pred, 'test_model')
    print("📊 Validation Results:")
    for metric, value in results.items():
        if isinstance(value, dict):
            print(f"   {metric}:")
            for k, v in value.items():
                print(f"     {k}: {v:.4f}")
        else:
            print(f"   {metric}: {value:.4f}")
    
    # Test model comparison
    models_predictions = {
        'model_1': y_true + np.random.randn(100) * 1,
        'model_2': y_true + np.random.randn(100) * 3
    }
    
    comparison_df = validator.compare_models(models_predictions, y_true)
    print("\n📈 Model Comparison:")
    print(comparison_df.round(4))
    
    print("✅ Model validator test completed!")
    return validator

if __name__ == "__main__":
    test_model_validator()