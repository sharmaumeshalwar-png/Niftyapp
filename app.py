import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🎯 Infinite Convergence: Resilience Mode [Batch Process]")

@st.cache_data(ttl=3600)
def get_batch_data():
    all_chunks = []
    end = datetime.now()
    # 2 saal ko 4-4 mahine ke 6 chunks mein baata (6 * 4 = 24 mahine)
    for i in range(6):
        start_chunk = end - timedelta(days=(i+1)*120)
        end_chunk = end - timedelta(days=i*120)
        
        chunk = yf.download("^NSEI", start=start_chunk, end=end_chunk, interval="1h", progress=False)
        if not chunk.empty:
            all_chunks.append(chunk)
            
    if not all_chunks:
        return pd.DataFrame()
        
    df = pd.concat(all_chunks).sort_index().ffill()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

df = get_batch_data()

if df.empty:
    st.error("Server Timeout: API se data phir bhi nahi mil raha. Check kijiye ki Nifty data available hai ya nahi.")
else:
    st.success(f"✅ Data Success: {len(df)} total candles load hui.")
    
    # 8-Candle Convergence Engine
    df['Convergence_Price'] = df['Close'].rolling(window=8).mean()
    df['Range_Low'] = df['Low'].rolling(window=8).min()
    df['Range_High'] = df['High'].rolling(window=8).max()
    df['Status'] = 'LOCK'
    
    # Audit View
    res_df = df.dropna()[['Convergence_Price', 'Range_Low', 'Range_High', 'Status']]
    st.dataframe(res_df.sort_index(ascending=False).head(50), use_container_width=True)
