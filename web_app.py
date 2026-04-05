# web_app.py - FIXED VERSION


# run tthis : python -m streamlit run web_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .stock-card {
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid;
        margin: 0.3rem;
        background: white;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .stock-card:hover {
        transform: translateY(-2px);
    }
    .bullish-card {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    }
    .moderate-card {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
    }
    .bearish-card {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
    }
    .positive {
        color: #059669;
        font-weight: bold;
        font-size: 1em;
        background: rgba(16, 185, 129, 0.1);
        padding: 2px 6px;
        border-radius: 6px;
    }
    .negative {
        color: #dc2626;
        font-weight: bold;
        font-size: 1em;
        background: rgba(239, 68, 68, 0.1);
        padding: 2px 6px;
        border-radius: 6px;
    }
    .neutral {
        color: #d97706;
        font-weight: bold;
        font-size: 1em;
        background: rgba(245, 158, 11, 0.1);
        padding: 2px 6px;
        border-radius: 6px;
    }
    .stock-symbol {
        color: #1f2937;
        font-weight: 700;
        font-size: 1.1em;
        margin: 0;
    }
    .stock-price {
        color: #111827;
        font-weight: 800;
        font-size: 1.3em;
        margin: 5px 0;
    }
    .stock-signal {
        color: #374151;
        font-weight: 600;
        font-size: 0.9em;
        margin: 3px 0;
    }
    .stock-time {
        color: #6b7280;
        font-size: 0.8em;
        margin: 0;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.3rem;
    }
    .last-update {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e9ecef;
        margin-top: 1rem;
    }
    .clock-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ALL 27 STOCKS
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'AMD', 'INTC', 
           'JPM', 'BAC', 'GS', 'MS', 'JNJ', 'PFE', 'UNH', 'MRK', 'XOM', 'CVX', 'COP', 
           'WMT', 'TGT', 'COST', 'SPY', 'QQQ', 'DIA']

def get_stock_data(symbol, period="1mo"):
    """Get stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        return hist
    except:
        return None

def calculate_technical_indicators(df):
    """Calculate basic technical indicators"""
    if df is None or len(df) < 20:
        return df
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    return df

def main():
    # Header with better styling
    st.markdown('<h1 class="main-header">🚀 AI Stock Predictor Pro</h1>', unsafe_allow_html=True)
    
    # Live Clock Display
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f'<div class="clock-display">🕒 LIVE MARKET DATA | {current_time}</div>', unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Controls")
        st.markdown("---")
        
        selected_symbols = st.multiselect(
            "**Select Stocks to Display**", 
            SYMBOLS, 
            default=SYMBOLS[:12]
        )
        
        chart_period = st.selectbox(
            "**Chart Period**",
            ["1mo", "3mo", "6mo", "1y"],
            index=1
        )
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            update_btn = st.button("🔄 **Refresh Now**", use_container_width=True)
        with col2:
            auto_update = st.checkbox("**Auto Update**", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.write(f"**Total Stocks:** {len(SYMBOLS)}")
        st.write(f"**Selected:** {len(selected_symbols)}")
        st.write(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">📊 Live Market Analysis</div>', unsafe_allow_html=True)
        
        if not selected_symbols:
            st.warning("🎯 Please select stocks from sidebar to begin analysis")
            return
        
        # Enhanced Stock cards with 3 color categories
        for i in range(0, len(selected_symbols), 4):
            cols = st.columns(4)
            for j, symbol in enumerate(selected_symbols[i:i+4]):
                with cols[j]:
                    try:
                        data = get_stock_data(symbol, "2d")
                        if data is not None and len(data) >= 2:
                            current_price = data['Close'].iloc[-1]
                            prev_price = data['Close'].iloc[-2]
                            change = ((current_price - prev_price) / prev_price) * 100
                            
                            # Enhanced 3-color logic
                            if change > 1:
                                color_class = "positive"
                                card_class = "bullish-card"
                                emoji = "🚀"
                                signal = "🟢 STRONG BUY"
                            elif change > 0:
                                color_class = "neutral"
                                card_class = "moderate-card"
                                emoji = "📈"
                                signal = "🟡 MODERATE BUY"
                            elif change > -1:
                                color_class = "neutral"
                                card_class = "moderate-card"
                                emoji = "📊"
                                signal = "🟡 HOLD"
                            else:
                                color_class = "negative"
                                card_class = "bearish-card"
                                emoji = "📉"
                                signal = "🔴 SELL"
                            
                            st.markdown(f"""
                            <div class="stock-card {card_class}">
                                <p class="stock-symbol">{symbol} {emoji}</p>
                                <p class="stock-price">${current_price:.2f}</p>
                                <p class="{color_class}">{change:+.2f}%</p>
                                <p class="stock-signal">{signal}</p>
                                <p class="stock-time">{datetime.now().strftime('%H:%M')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error fetching {symbol}")
    
    with col2:
        st.markdown('<div class="section-header">📈 Advanced Charts</div>', unsafe_allow_html=True)
        
        if selected_symbols:
            chart_symbol = st.selectbox("**Select Stock:**", selected_symbols)
            
            chart_data = get_stock_data(chart_symbol, chart_period)
            if chart_data is not None:
                tab1, tab2 = st.tabs(["📊 Candlestick", "📈 Line Chart"])
                
                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close'],
                        name='Price'
                    ))
                    
                    if len(chart_data) >= 50:
                        chart_data = calculate_technical_indicators(chart_data)
                        fig.add_trace(go.Scatter(
                            x=chart_data.index,
                            y=chart_data['SMA_20'],
                            name='SMA 20',
                            line=dict(color='orange', width=2, dash='dash')
                        ))
                    
                    fig.update_layout(
                        title=f"<b>{chart_symbol} - Price Chart</b>",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                        showlegend=True,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Scatter(
                        x=chart_data.index,
                        y=chart_data['Close'],
                        name='Close Price',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    fig_line.update_layout(
                        title=f"<b>{chart_symbol} - Price Trend</b>",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
        
        # Enhanced Market Summary
        st.markdown('<div class="section-header">📋 Market Intelligence</div>', unsafe_allow_html=True)
        
        if selected_symbols:
            bullish_count = 0
            moderate_count = 0
            bearish_count = 0
            total_change = 0
            
            for symbol in selected_symbols:
                try:
                    data = get_stock_data(symbol, "2d")
                    if data is not None and len(data) >= 2:
                        current_price = data['Close'].iloc[-1]
                        prev_price = data['Close'].iloc[-2]
                        change = ((current_price - prev_price) / prev_price) * 100
                        total_change += change
                        
                        if change > 1:
                            bullish_count += 1
                        elif change > -1:
                            moderate_count += 1
                        else:
                            bearish_count += 1
                            
                except:
                    pass
            
            if len(selected_symbols) > 0:
                avg_change = total_change / len(selected_symbols)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🚀 Strong Bullish", f"{bullish_count}")
                    st.metric("📊 Market Average", f"{avg_change:+.2f}%")
                with col2:
                    st.metric("📈 Moderate", f"{moderate_count}")
                    st.metric("📉 Bearish", f"{bearish_count}")
    
    # Footer with refresh info
    st.markdown("---")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f"""
    <div class="last-update">
        <h4>🕒 Last Update: {current_time}</h4>
        <p>Next auto-refresh in 30 seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Refresh logic
    if auto_update:
        time.sleep(30)
        st.rerun()
    
    if update_btn:
        st.rerun()

if __name__ == "__main__":
    main()