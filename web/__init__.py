"""
Web Module
==========

This module provides the web interface for the AI Stock Predictor including:
- Real-time dashboard
- REST API endpoints
- WebSocket live updates
- Interactive visualizations
"""

from .dashboard.app import app, socketio
from .api.routes import api

# Register blueprints
app.register_blueprint(api, url_prefix='/api')

__version__ = "1.0.0"
__author__ = "Stock Predictor Team"

__all__ = ['app', 'socketio', 'api']

print(f"✅ Web module v{__version__} loaded successfully!")
print("Available components:")
print("  - Flask Application: Web server and routing")
print("  - SocketIO: Real-time WebSocket communications") 
print("  - REST API: JSON API endpoints for mobile/app access")
print("  - Dashboard: Interactive web interface with Plotly charts")