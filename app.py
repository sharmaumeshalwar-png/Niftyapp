import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 Nifty 50: Max 2-Year Full-Candle Engine [50:50 Split]")

# =====================================================================
# CORE MATHEMATICAL FUNCTIONS
# =====================================================================
def apply_kalman(data, initial_p=100.0, q=0.0001, r=2.5):
    x = data[0]; p = initial_p; filtered = []
    for z in data:
        p += q; k = p / (p + r); x += k * (z - x); p *= (1 - k)
        filtered.append(x)
    return np.array(filtered)

def apply_step_momentum(data):
    x = data[0]; p = 1.0; q = 0.05; r = 0.2; filtered = []
    for z in data:
        p += q; k = p / (p + r); x += k * (z - x); p *= (1 - k)
        filtered.append(x)
    return np.round(filtered)

# =====================================================================
# DATA PIPELINE (Max Candles Fetch)
# =====================================================================
@st.cache_data
def get_data():
    # 2 years = 730 days
    end = datetime.now()
    start = end - timedelta(days=730)
    raw = yf.download("^NSEI", start=start, end=end, interval="1h", auto_adjust=True)
    return raw.ffill().dropna()

with st.spinner("Processing Maximum Candles (1H) for 2-Year Horizon..."):
    df = get_data()
    total_candles = len(df)
    
    # 1. Base Kalman Price
    df['Kalman_Price'] = apply_kalman(df['Close'].values)
    
    # 2. Weighted Momentum
    df['Raw_WM'] = df['Close'] - df['Kalman_Price']
    df['Weighted_Momentum'] = apply_kalman(df['Raw_WM'].values, initial_p=0.5, q=0.01, r=0.5)
    
    # 3. Step Momentum
    df['Step_Momentum'] = apply_step_momentum(df['Weighted_Momentum'].values)
    
    # 4. Accumulator (Trend Lock)
    # Score increases if Weighted Momentum > 0 and Step Momentum > 0
    df['Accumulator'] = 0
    signal = 0
    for i in range(len(df)):
        if df['Weighted_Momentum'].iloc[i] > 0 and df['Step_Momentum'].iloc[i] > 0:
            signal = min(signal + 1, 5)
        elif df['Weighted_Momentum'].iloc[i] < 0 and df['Step_Momentum'].iloc[i] < 0:
            signal = max(signal - 1, -5)
        df.iloc[i, df.columns.get_loc('Accumulator')] = signal

    # 5. 50:50 Split (Displaying the Second Half)
    split_idx = int(total_candles * 0.50)
    display_df = df.iloc[split_idx:].sort_index(ascending=False)

    st.write(f"### 📈 Total Candles Processed: {total_candles} | Live Prediction Window (Last 50%): {len(display_df)} Rows")
    
    st.dataframe(display_df[['Close', 'Weighted_Momentum', 'Step_Momentum', 'Accumulator']], 
                 use_container_width=True)
