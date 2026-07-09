import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

st.set_page_config(layout="wide")
st.title("🎯 Infinite Convergence: 2-Year Full Data Sniper")

@st.cache_data(ttl=3600)
def get_full_2year_data():
    # Aaj ki date se 2 saal pehle tak ka data
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=730)
    
    # 2-Year data request (Explicit dates)
    df = yf.download("^NSEI", start=start_date, end=end_date, interval="1h", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df.ffill().dropna()

# Execution
df = get_full_2year_data()
st.write(f"📊 Total Data Points Loaded: {len(df)}") # Check karne ke liye

# 8-Candle window logic
results = []
for i in range(8, len(df)):
    window = df.iloc[i-8:i]
    low, high = window['Low'].min(), window['High'].max()
    conv_price = window['Close'].mean()
    
    results.append({
        'Target_Time': df.index[i],
        'Convergence_Price': round(conv_price, 2),
        'Range_Low': round(low, 2),
        'Range_High': round(high, 2),
        'Status': 'LOCK'
    })

data = pd.DataFrame(results)

st.subheader("📋 2-Year Audit (Infinite Convergence Lock)")
st.dataframe(data.sort_index(ascending=False).head(50), use_container_width=True)
