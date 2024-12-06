import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Expanded Indices and Stocks Options
indices_options = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK", 
    "Sensex": "^BSESN",
    "BSE Midcap": "^BSEME",
    "Nifty IT": "^NIFTYIT"
}

stocks_options = {
    "Reliance": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "State Bank of India": "SBIN.NS"
}

# Time interval options
intervals = {
    "1 day": "1d",
    "1 week": "1wk", 
    "1 month": "1mo"
}

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    # Calculate EMAs
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['20dSTD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['20dSTD'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['20dSTD'] * 2)
    
    return df

def get_trading_signals(df):
    signals = []
    
    # EMA Signal
    if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]:
        signals.append(("EMA Cross", "BULLISH", "Short-term EMA above long-term EMA"))
    else:
        signals.append(("EMA Cross", "BEARISH", "Short-term EMA below long-term EMA"))
    
    # RSI Signal
    last_rsi = df['RSI'].iloc[-1]
    if last_rsi > 70:
        signals.append(("RSI", "OVERBOUGHT", f"RSI at {last_rsi:.2f}"))
    elif last_rsi < 30:
        signals.append(("RSI", "OVERSOLD", f"RSI at {last_rsi:.2f}"))
    else:
        signals.append(("RSI", "NEUTRAL", f"RSI at {last_rsi:.2f}"))
    
    # MACD Signal
    if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1]:
        signals.append(("MACD", "BULLISH", "MACD above signal line"))
    else:
        signals.append(("MACD", "BEARISH", "MACD below signal line"))
    
    # Bollinger Bands Signal
    last_close = df['Close'].iloc[-1]
    if last_close > df['Upper_Band'].iloc[-1]:
        signals.append(("Bollinger Bands", "OVERBOUGHT", "Price above upper band"))
    elif last_close < df['Lower_Band'].iloc[-1]:
        signals.append(("Bollinger Bands", "OVERSOLD", "Price below lower band"))
    else:
        signals.append(("Bollinger Bands", "NEUTRAL", "Price within bands"))
        
    return signals

def get_technical_analysis(price_data):
    try:
        # Calculate basic metrics
        current_price = float(price_data['Close'].iloc[-1])
        opening_price = float(price_data['Open'].iloc[0])
        price_change = ((current_price - opening_price) / opening_price) * 100
        volume = float(price_data['Volume'].mean())
        
        # Calculate technical indicators
        df_indicators = calculate_technical_indicators(price_data.copy())
        
        # Get trading signals
        signals = get_trading_signals(df_indicators)
        
        # Generate analysis text
        analysis = f"""
### Price Analysis
- Current Price: ₹{current_price:,.2f}
- Price Change: {price_change:,.2f}%
- Average Volume: {volume:,.0f}

### Technical Indicators
- EMA (20): ₹{df_indicators['EMA20'].iloc[-1]:,.2f}
- EMA (50): ₹{df_indicators['EMA50'].iloc[-1]:,.2f}
- RSI (14): {df_indicators['RSI'].iloc[-1]:.2f}
- MACD: {df_indicators['MACD'].iloc[-1]:.2f}

### Trading Signals
"""
        
        # Add signals to analysis
        for signal_type, signal, description in signals:
            color = ""
            if signal in ["BULLISH", "OVERSOLD"]:
                color = "green"
            elif signal in ["BEARISH", "OVERBOUGHT"]:
                color = "red"
            else:
                color = "orange"
                
            analysis += f"- {signal_type}: <span style='color: {color}'>{signal}</span> - {description}\n"
        
        return analysis, signals
        
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")
        return "Error generating analysis", []

def analyze_news_sentiment(symbol):
    try:
        # Get stock info
        stock = yf.Ticker(symbol)
        news = stock.news
        
        if not news:
            return "NEUTRAL", []
            
        # Initialize VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
            
        # Analyze sentiment for each news item
        sentiments = []
        compound_scores = []
        
        for item in news[:5]:  # Analyze last 5 news items
            sentiment = analyzer.polarity_scores(item['title'])
            compound_score = sentiment['compound']
            compound_scores.append(compound_score)
            
            # Determine sentiment label
            if compound_score >= 0.05:
                sentiment_label = "POSITIVE"
            elif compound_score <= -0.05:
                sentiment_label = "NEGATIVE"
            else:
                sentiment_label = "NEUTRAL"
            
            sentiments.append({
                'title': item['title'],
                'sentiment': sentiment_label,
                'score': compound_score
            })
        
        # Calculate overall sentiment
        avg_sentiment = sum(compound_scores) / len(compound_scores)
        if avg_sentiment >= 0.05:
            overall_sentiment = "BULLISH 📈"
        elif avg_sentiment <= -0.05:
            overall_sentiment = "BEARISH 📉"
        else:
            overall_sentiment = "NEUTRAL ↔️"
            
        return overall_sentiment, sentiments
        
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return "NEUTRAL", []

def fetch_stock_data(symbol, interval):
    try:
        # Fetch data with appropriate period and interval
        ticker = yf.Ticker(symbol)
        
        # Use history method instead of download
        data = ticker.history(
            period="1y",  # Fetch 1 year of data
            interval=interval
        )
        
        # Check if data is empty
        if data.empty:
            st.warning(f"No data available for {symbol} at {interval} interval.")
            return None
        
        # Reset index to make date a column
        data = data.reset_index()
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_enhanced_candlestick_chart(data, selected_option, symbol):
    try:
        # Ensure we have the correct columns
        if not all(col in data.columns for col in ['Date', 'Open', 'High', 'Low', 'Close']):
            st.error("Missing required price data columns")
            return None

        # Calculate technical indicators
        df_with_indicators = calculate_technical_indicators(data.copy())

        # Create figure with candlesticks and indicators
        fig = go.Figure(data=[
            go.Candlestick(
                x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlesticks'
            ),
            # Add EMAs
            go.Scatter(
                x=data['Date'], 
                y=df_with_indicators['EMA20'], 
                mode='lines', 
                name='EMA 20',
                line=dict(color='blue', width=2)
            ),
            go.Scatter(
                x=data['Date'], 
                y=df_with_indicators['EMA50'], 
                mode='lines', 
                name='EMA 50',
                line=dict(color='red', width=2)
            ),
            # Add Bollinger Bands
            go.Scatter(
                x=data['Date'],
                y=df_with_indicators['Upper_Band'],
                mode='lines',
                name='Upper BB',
                line=dict(color='gray', dash='dash')
            ),
            go.Scatter(
                x=data['Date'],
                y=df_with_indicators['Lower_Band'],
                mode='lines',
                name='Lower BB',
                line=dict(color='gray', dash='dash')
            )
        ])

        # Customize layout
        fig.update_layout(
            title=f"{selected_option} Price Movement",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=600
        )

        return fig, df_with_indicators
    
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None, None

def main():
    st.title("📈 Advanced Stock and Index Tracker")
    st.sidebar.header("Options")

    # User selection
    data_type = st.sidebar.radio("Select Data Type", ["Indices", "Stocks"])
    options = indices_options if data_type == "Indices" else stocks_options
    selected_option = st.sidebar.selectbox("Choose", list(options.keys()))
    symbol = options[selected_option]

    # Time interval selection
    selected_interval = st.sidebar.selectbox("Select Time Interval", list(intervals.keys()))

    # Fetch data
    data = fetch_stock_data(symbol, intervals[selected_interval])

    if data is not None and not data.empty:
        st.success(f"Data Loaded Successfully for {selected_option}")

        # Create and display enhanced candlestick chart
        chart, df_with_indicators = create_enhanced_candlestick_chart(data, selected_option, symbol)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

        # Create columns for detailed analysis
        col1, col2 = st.columns(2)

        with col1:
            # Technical Analysis
            with st.expander("📊 Technical Analysis", expanded=True):
                if df_with_indicators is not None:
                    analysis, signals = get_technical_analysis(data)
                    st.markdown(analysis, unsafe_allow_html=True)
                else:
                    st.error("Could not generate technical analysis")

        with col2:
            # News Sentiment Analysis
            with st.expander("📰 News Sentiment Analysis", expanded=True):
                sentiment, news_sentiments = analyze_news_sentiment(symbol)
                st.markdown(f"### Overall Market Sentiment: {sentiment}")
                
                if isinstance(news_sentiments, list) and news_sentiments:
                    for item in news_sentiments:
                        color = "green" if item['sentiment'] == "POSITIVE" else "red"
                        st.markdown(f"""
                            <div style='padding: 10px; border-left: 3px solid {color};'>
                                <p>{item['title']}</p>
                                <p style='color: {color};'>Sentiment: {item['sentiment']} ({item['score']:.2f})</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No recent news available for analysis")

        # Additional Market Insights
        st.subheader("Market Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            last_close = float(data['Close'].iloc[-1]) if not data.empty else None
            st.metric("Last Close", f"₹{last_close:.2f}" if last_close is not None else "N/A")
        
        with col2:
            if len(data) > 1:
                current_close = float(data['Close'].iloc[-1])
                previous_close = float(data['Close'].iloc[-2])
                daily_change = ((current_close - previous_close) / previous_close) * 100
                st.metric("Daily Change", f"{daily_change:.2f}%", 
                          "▲" if daily_change >= 0 else "▼")
            else:
                st.metric("Daily Change", "N/A")
        
        with col3:
            last_volume = float(data['Volume'].iloc[-1]) if not data.empty else None
            st.metric("Trading Volume", f"{last_volume:,.0f}" if last_volume is not None else "N/A")

    else:
        st.error("No data could be retrieved. Please try another selection.")

# Run the app
if __name__ == "__main__":
    main()