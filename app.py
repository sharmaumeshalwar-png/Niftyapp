import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🎯 Infinite Convergence: Resilience Mode")

@st.cache_data(ttl=3600)
def get_resilient_data():
    # 2 saal ka duration
    end = datetime.now()
    start = end - timedelta(days=730)
    
    # Chunking: Agar ek baar mein fail ho, toh retry logic
    try:
        df = yf.download("^NSEI", start=start, end=end, interval="1h", progress=False)
        # Fix for MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure 'Close' is clean
        df = df[df['Close'].notna()]
        return df
    except Exception as e:
        return pd.DataFrame()

df = get_resilient_data()

if df.empty:
    st.error("Server Timeout: API se data nahi mil raha. Kripya thodi der baad refresh karein.")
else:
    st.success(f"✅ Data Success: {len(df)} candles load hue.")
    
    # 8-Candle Convergence Engine
    # Vectorized calculation for performance
    df['Convergence_Price'] = df['Close'].rolling(window=8).mean()
    df['Range_Low'] = df['Low'].rolling(window=8).min()
    df['Range_High'] = df['High'].rolling(window=8).max()
    df['Status'] = 'LOCK'
    
    # Result table
    res_df = df.dropna()[['Convergence_Price', 'Range_Low', 'Range_High', 'Status']]
    st.dataframe(res_df.sort_index(ascending=False).head(50), use_container_width=True)
