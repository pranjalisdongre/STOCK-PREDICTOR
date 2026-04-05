import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PerformanceMonitor:
    def __init__(self):
        self.performance_history = {}
        self.model_drift_alerts = []
    
    def track_model_performance(self, model_name, metrics, timestamp=None):
        """Track model performance over time"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        record = {
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        self.performance_history[model_name].append(record)
        
        # Keep only last 100 records per model
        if len(self.performance_history[model_name]) > 100:
            self.performance_history[model_name] = self.performance_history[model_name][-100:]
    
    def detect_model_drift(self, model_name, window_size=20, threshold=0.1):
        """Detect model performance drift"""
        if model_name not in self.performance_history:
            return None
        
        records = self.performance_history[model_name]
        
        if len(records) < window_size * 2:
            return None
        
        # Split into recent and historical windows
        recent_records = records[-window_size:]
        historical_records = records[-(window_size*2):-window_size]
        
        # Calculate average MAE for both periods
        recent_mae = np.mean([r['metrics'].get('mae', 0) for r in recent_records])
        historical_mae = np.mean([r['metrics'].get('mae', 0) for r in historical_records])
        
        # Calculate drift
        if historical_mae > 0:
            drift_ratio = (recent_mae - historical_mae) / historical_mae
        else:
            drift_ratio = 0
        
        # Check if drift exceeds threshold
        if abs(drift_ratio) > threshold:
            alert = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'drift_ratio': drift_ratio,
                'recent_mae': recent_mae,
                'historical_mae': historical_mae,
                'severity': 'high' if abs(drift_ratio) > 0.2 else 'medium'
            }
            
            self.model_drift_alerts.append(alert)
            return alert
        
        return None
    
    def get_performance_trends(self, model_name, metric='mae'):
        """Get performance trends for a specific metric"""
        if model_name not in self.performance_history:
            return None
        
        records = self.performance_history[model_name]
        
        timestamps = [r['timestamp'] for r in records]
        values = [r['metrics'].get(metric, 0) for r in records]
        
        trend_data = pd.DataFrame({
            'timestamp': timestamps,
            metric: values
        })
        
        # Calculate moving averages
        trend_data['ma_7'] = trend_data[metric].rolling(7).mean()
        trend_data['ma_30'] = trend_data[metric].rolling(30).mean()
        
        return trend_data
    
    def generate_performance_report(self, days=30):
        """Generate performance report for the last N days"""
        report = {
            'generated_at': datetime.now(),
            'summary': {},
            'alerts': [],
            'recommendations': []
        }
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for model_name, records in self.performance_history.items():
            # Filter recent records
            recent_records = [r for r in records if r['timestamp'] >= cutoff_date]
            
            if not recent_records:
                continue
            
            # Calculate summary statistics
            maes = [r['metrics'].get('mae', 0) for r in recent_records]
            r2_scores = [r['metrics'].get('r2', 0) for r in recent_records]
            direction_accuracies = [r['metrics'].get('direction_accuracy', 0) for r in recent_records]
            
            report['summary'][model_name] = {
                'avg_mae': np.mean(maes),
                'avg_r2': np.mean(r2_scores),
                'avg_direction_accuracy': np.mean(direction_accuracies),
                'records_count': len(recent_records),
                'latest_timestamp': max(r['timestamp'] for r in recent_records)
            }
        
        # Add recent alerts
        recent_alerts = [alert for alert in self.model_drift_alerts 
                        if alert['timestamp'] >= cutoff_date]
        report['alerts'] = recent_alerts
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report['summary'])
        
        return report
    
    def _generate_recommendations(self, summary):
        """Generate recommendations based on performance"""
        recommendations = []
        
        for model_name, stats in summary.items():
            if stats['avg_mae'] > 10: 