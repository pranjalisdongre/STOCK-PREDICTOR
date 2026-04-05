// Dashboard JavaScript
class StockDashboard {
    constructor() {
        this.socket = null;
        this.currentSymbol = 'AAPL';
        this.portfolioData = null;
        this.init();
    }

    init() {
        this.connectWebSocket();
        this.loadInitialData();
        this.setupEventListeners();
        this.startLiveUpdates();
    }

    connectWebSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus(true);
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
        });

        this.socket.on('stock_update', (data) => {
            this.handleStockUpdate(data);
        });

        this.socket.on('portfolio_update', (data) => {
            this.handlePortfolioUpdate(data);
        });

        this.socket.on('market_overview', (data) => {
            this.handleMarketOverview(data);
        });
    }

    updateConnectionStatus(connected) {
        const indicator = document.getElementById('live-indicator');
        if (connected) {
            indicator.className = 'badge bg-success';
            indicator.innerHTML = '<i class="fas fa-circle"></i> LIVE';
        } else {
            indicator.className = 'badge bg-danger';
            indicator.innerHTML = '<i class="fas fa-circle"></i> OFFLINE';
        }
    }

    async loadInitialData() {
        await this.loadMarketOverview();
        await this.loadPortfolioData();
        await this.loadStockData(this.currentSymbol);
        await this.loadPredictions();
        await this.loadRecentTrades();
    }

    async loadMarketOverview() {
        try {
            const response = await fetch('/api/market-overview');
            const data = await response.json();
            this.updateMarketOverview(data);
        } catch (error) {
            console.error('Error loading market overview:', error);
        }
    }

    async loadPortfolioData() {
        try {
            const response = await fetch('/api/portfolio');
            this.portfolioData = await response.json();
            this.updatePortfolioDisplay(this.portfolioData);
            this.createPortfolioChart();
        } catch (error) {
            console.error('Error loading portfolio data:', error);
        }
    }

    async loadStockData(symbol) {
        try {
            const response = await fetch(`/api/stock-data/${symbol}`);
            const data = await response.json();
            this.createStockChart(data);
            this.loadTradingSignals(symbol);
        } catch (error) {
            console.error('Error loading stock data:', error);
        }
    }

    async loadPredictions() {
        try {
            const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'];
            const predictions = [];
            
            for (const symbol of symbols) {
                const response = await fetch(`/api/predictions/${symbol}`);
                const data = await response.json();
                if (data) {
                    predictions.push(data);
                }
            }
            
            this.updatePredictionsTable(predictions);
        } catch (error) {
            console.error('Error loading predictions:', error);
        }
    }

    async loadRecentTrades() {
        try {
            // This would come from your portfolio API
            // For now, we'll simulate some data
            const simulatedTrades = [
                {
                    date: new Date().toISOString().split('T')[0],
                    symbol: 'AAPL',
                    action: 'BUY',
                    quantity: 10,
                    price: 150.25,
                    pnl: '+$45.50',
                    status: 'EXECUTED'
                },
                {
                    date: new Date(Date.now() - 86400000).toISOString().split('T')[0],
                    symbol: 'TSLA',
                    action: 'SELL', 
                    quantity: 5,
                    price: 245.80,
                    pnl: '+$123.25',
                    status: 'EXECUTED'
                }
            ];
            
            this.updateRecentTrades(simulatedTrades);
        } catch (error) {
            console.error('Error loading recent trades:', error);
        }
    }

    async loadTradingSignals(symbol) {
        try {
            const response = await fetch('/api/trading-signal', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ symbol: symbol })
            });
            
            const signal = await response.json();
            this.updateTradingSignals(symbol, signal);
        } catch (error) {
            console.error('Error loading trading signals:', error);
        }
    }

    updateMarketOverview(data) {
        // Update market sentiment
        const sentimentEl = document.getElementById('market-sentiment');
        const sentimentTrendEl = document.getElementById('sentiment-trend');
        
        if (data.market_sentiment) {
            sentimentEl.textContent = data.market_sentiment.trend;
            sentimentTrendEl.textContent = `Score: ${data.market_sentiment.score.toFixed(2)}`;
            
            if (data.market_sentiment.trend === 'BULLISH') {
                sentimentEl.className = 'value positive';
            } else {
                sentimentEl.className = 'value negative';
            }
        }

        // Update AI confidence
        const confidenceEl = document.getElementById('ai-confidence');
        confidenceEl.textContent = data.market_sentiment ? 
            `${Math.round(data.market_sentiment.confidence * 100)}%` : '--%';
    }

    updatePortfolioDisplay(data) {
        if (!data.portfolio_summary) return;

        const summary = data.portfolio_summary;
        const performance = data.performance || {};

        // Portfolio value
        const portfolioValueEl = document.getElementById('portfolio-value');
        portfolioValueEl.textContent = `$${summary.portfolio_value?.toLocaleString() || '---'}`;

        // Daily P&L (simulated)
        const dailyPnlEl = document.getElementById('daily-pnl');
        const dailyPnlPercentEl = document.getElementById('daily-pnl-percent');
        const dailyChange = (Math.random() - 0.5) * 200;
        const dailyPercent = (dailyChange / (summary.portfolio_value || 10000)) * 100;
        
        dailyPnlEl.textContent = `$${dailyChange.toFixed(2)}`;
        dailyPnlPercentEl.textContent = `${dailyPercent.toFixed(2)}%`;
        
        if (dailyChange >= 0) {
            dailyPnlEl.className = 'value positive';
            dailyPnlPercentEl.className = 'positive';
        } else {
            dailyPnlEl.className = 'value negative';
            dailyPnlPercentEl.className = 'negative';
        }

        // Active positions
        const positionsEl = document.getElementById('active-positions');
        positionsEl.textContent = summary.open_positions || '0';

        // Win rate
        const winRateEl = document.getElementById('win-rate');
        winRateEl.textContent = performance.win_rate ? 
            `${Math.round(performance.win_rate * 100)}%` : '--%';
    }

    createStockChart(data) {
        if (!data.chart_data) return;

        const chartData = JSON.parse(data.chart_data);
        Plotly.newPlot('stock-chart', chartData.data, chartData.layout, {
            responsive: true,
            displayModeBar: true
        });
    }

    createPortfolioChart() {
        // Simulated portfolio performance data
        const dates = [];
        const values = [];
        let currentValue = 10000;
        
        for (let i = 30; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            dates.push(date.toISOString().split('T')[0]);
            
            // Random walk for portfolio value
            currentValue += (Math.random() - 0.5) * 200;
            values.push(Math.max(8000, currentValue));
        }

        const trace = {
            x: dates,
            y: values,
            type: 'scatter',
            mode: 'lines',
            name: 'Portfolio Value',
            line: { color: '#667eea', width: 3 }
        };

        const layout = {
            height: 300,
            margin: { t: 30, r: 30, b: 30, l: 50 },
            showlegend: false,
            xaxis: { 
                showgrid: false,
                tickformat: '%b %d'
            },
            yaxis: {
                tickprefix: '$',
                showgrid: true,
                gridcolor: '#f0f0f0'
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };

        Plotly.newPlot('portfolio-chart', [trace], layout);
    }

    updateTradingSignals(symbol, signal) {
        const container = document.getElementById('trading-signals-container');
        
        if (!signal.signal) {
            container.innerHTML = '<div class="alert alert-info">No signal available</div>';
            return;
        }

        const signalClass = this.getSignalClass(signal.signal);
        const recommendation = signal.recommendation || 'No recommendation';
        
        container.innerHTML = `
            <div class="alert ${signalClass}">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h5 class="alert-heading mb-1">${symbol} - ${signal.signal}</h5>
                        <p class="mb-0">${recommendation}</p>
                        <small>Confidence: ${Math.round(signal.confidence * 100)}%</small>
                    </div>
                    <div class="text-end">
                        <div class="h4 mb-0">
                            ${this.getSignalIcon(signal.signal)}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    updatePredictionsTable(predictions) {
        const tbody = document.getElementById('predictions-table');
        
        if (!predictions || predictions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center">No predictions available</td></tr>';
            return;
        }

        tbody.innerHTML = predictions.map(pred => {
            const current = pred.current_price?.toFixed(2) || '--';
            const predicted = pred.prediction?.ensemble_prediction?.toFixed(2) || '--';
            const signal = pred.trading_signal || 'HOLD';
            const confidence = pred.prediction?.confidence ? 
                Math.round(pred.prediction.confidence * 100) : '--';
            
            return `
                <tr>
                    <td><strong>${pred.symbol}</strong></td>
                    <td>$${current}</td>
                    <td>$${predicted}</td>
                    <td>
                        <span class="badge ${this.getSignalBadgeClass(signal)}">
                            ${signal}
                        </span>
                    </td>
                    <td>${confidence}%</td>
                </tr>
            `;
        }).join('');
    }

    updateRecentTrades(trades) {
        const tbody = document.getElementById('recent-trades');
        
        tbody.innerHTML = trades.map(trade => {
            const pnlClass = trade.pnl?.includes('+') ? 'positive' : 'negative';
            
            return `
                <tr>
                    <td>${trade.date}</td>
                    <td><strong>${trade.symbol}</strong></td>
                    <td>
                        <span class="badge ${trade.action === 'BUY' ? 'bg-success' : 'bg-danger'}">
                            ${trade.action}
                        </span>
                    </td>
                    <td>${trade.quantity}</td>
                    <td>$${trade.price}</td>
                    <td class="${pnlClass}">${trade.pnl}</td>
                    <td>
                        <span class="badge bg-success">${trade.status}</span>
                    </td>
                </tr>
            `;
        }).join('');
    }

    getSignalClass(signal) {
        const classes = {
            'BUY': 'signal-buy',
            'SELL': 'signal-sell',
            'WEAK_BUY': 'signal-buy',
            'WEAK_SELL': 'signal-sell',
            'HOLD': 'signal-hold'
        };
        return classes[signal] || 'signal-hold';
    }

    getSignalBadgeClass(signal) {
        const classes = {
            'BUY': 'bg-success',
            'SELL': 'bg-danger', 
            'WEAK_BUY': 'bg-success',
            'WEAK_SELL': 'bg-danger',
            'HOLD': 'bg-warning text-dark'
        };
        return classes[signal] || 'bg-secondary';
    }

    getSignalIcon(signal) {
        const icons = {
            'BUY': '📈',
            'SELL': '📉',
            'WEAK_BUY': '↗️',
            'WEAK_SELL': '↘️',
            'HOLD': '⏸️'
        };
        return icons[signal] || '⏸️';
    }

    handleStockUpdate(data) {
        console.log('Stock update:', data);
        // Update real-time stock display
    }

    handlePortfolioUpdate(data) {
        console.log('Portfolio update:', data);
        this.updatePortfolioDisplay(data);
    }

    handleMarketOverview(data) {
        console.log('Market overview:', data);
        this.updateMarketOverview(data);
    }

    setupEventListeners() {
        // Stock selector
        const stockSelector = document.getElementById('stock-selector');
        const signalStockSelector = document.getElementById('signal-stock-selector');
        
        if (stockSelector) {
            stockSelector.addEventListener('change', (e) => {
                this.currentSymbol = e.target.value;
                this.loadStockData(this.currentSymbol);
            });
        }
        
        if (signalStockSelector) {
            signalStockSelector.addEventListener('change', (e) => {
                this.loadTradingSignals(e.target.value);
            });
        }

        // Refresh button
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadInitialData();
            });
        }

        // Export trades button
        const exportBtn = document.getElementById('export-trades');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.exportTrades();
            });
        }
    }

    exportTrades() {
        // Simple CSV export simulation
        const csvContent = "data:text/csv;charset=utf-8," 
            + "Date,Symbol,Action,Quantity,Price,P&L,Status\n"
            + "2024-01-15,AAPL,BUY,10,150.25,+45.50,EXECUTED\n"
            + "2024-01-14,TSLA,SELL,5,245.80,+123.25,EXECUTED";
        
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "trading_history.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    startLiveUpdates() {
        // Update time every second
        setInterval(() => {
            const timeEl = document.getElementById('current-time');
            if (timeEl) {
                timeEl.textContent = new Date().toLocaleTimeString();
            }
        }, 1000);

        // Refresh data every 30 seconds
        setInterval(() => {
            this.loadMarketOverview();
            this.loadPortfolioData();
        }, 30000);
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    window.stockDashboard = new StockDashboard();
});