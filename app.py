import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Nifty Custom 1-Hour System", layout="wide")
st.title("📊 Nifty 1-Hour Custom Algorithmic Dashboard")
st.write("Formula Applied: Column B Multiplier = 0.0001 | Timeframe: 1-Hour Candle")

# Fetch 1-Hour Nifty Data
@st.cache_data(ttl=300)
def load_data():
    # Fetching 1 month of hourly data
    df = yf.download(tickers="^NSEI", period="1mo", interval="1h")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

df = load_data()

if not df.empty:
    df = df.reset_index()
    
    # 1. Column D: Date and Time formatting (e.g., '03 May 10:15')
    # Handling yfinance datetime column name which is usually 'Datetime'
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Column D'] = pd.to_datetime(df[time_col]).dt.strftime('%d %b %H:%M')
    
    # 2. Column A: (High + Low) / 2
    df['Column A'] = (df['High'] + df['Low']) / 2
    
    # 3. Column B: Running Excel formula with 0.0001 multiplier
    multiplier = 0.0001
    col_b = np.zeros(len(df))
    
    if len(df) > 0:
        col_b[0] = df['Column A'].iloc[0]  # First row: B1 = A1
        
    for i in range(1, len(df)):
        a_current = df['Column A'].iloc[i]
        b_prev = col_b[i-1]
        col_b[i] = b_prev + (multiplier * (a_current - b_prev))
        
    df['Column B'] = col_b
    
    # 4. Column C: A - B
    df['Column C'] = df['Column A'] - df['Column B']
    
    # Top Metrics Display
    latest_a = float(df['Column A'].iloc[-1])
    latest_b = float(df['Column B'].iloc[-1])
    latest_c = float(df['Column C'].iloc[-1])
    latest_time = str(df['Column D'].iloc[-1])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Candle Time (Col D)", latest_time)
    col2.metric("Column A (H+L)/2", f"₹ {latest_a:.2f}")
    col3.metric("Column B (Running)", f"₹ {latest_b:.4f}")
    col4.metric("Column C (A - B)", f"{latest_c:.4f}")
    
    st.subheader("📋 System Logs - Custom Mathematical Table")
    
    # Rearranging columns for display order: D, A, B, C
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C']].copy()
    
    # Reverse to see latest candles on top
    show_df = show_df.iloc[::-1]
    
    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 
        'Column B': '{:.4f}', 
        'Column C': '{:.4f}'
    }), use_container_width=True)
else:
    st.error("Market data load karne mein dikkat aa rahi hai.")
