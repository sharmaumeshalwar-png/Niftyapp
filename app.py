import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🎯 Backtest: Price Convergence Audit (2-Year)")

@st.cache_data(ttl=3600)
def get_verified_data():
    # 2 saal ka data ek saath mangne ke bajaye, 1-1 saal karke handle karte hain
    end = datetime.now()
    start = end - timedelta(days=730)
    # 1h interval par stability ke liye thoda chota request
    df = yf.download("^NSEI", start=start, end=end, interval="1h", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df.ffill().dropna()

df = get_verified_data()

if df.empty:
    st.error("Data load nahi hua. Server busy hai.")
else:
    # 8-Hour Lock Logic
    df['LOCK'] = df['Close'].rolling(window=8).mean()
    
    # Audit: Kya price ne kabhi LOCK ko touch kiya?
    # Logic: Agar High >= LOCK aur Low <= LOCK, toh convergence 'HIT' hua
    df['Convergence_Status'] = 'MISS'
    mask = (df['High'] >= df['LOCK']) & (df['Low'] <= df['LOCK'])
    df.loc[mask, 'Convergence_Status'] = 'HIT'
    
    # Audit Table
    st.write(f"📊 Audit Results: {len(df)} candles checked.")
    st.dataframe(df[['Close', 'LOCK', 'Convergence_Status']].sort_index(ascending=False).head(50), use_container_width=True)

    # Success Rate calculation
    hit_rate = (df['Convergence_Status'] == 'HIT').mean() * 100
    st.metric("Total Convergence Success Rate", f"{hit_rate:.2f}%")
