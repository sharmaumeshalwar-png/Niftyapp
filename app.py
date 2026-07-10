import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 Nifty 50: 2-Year Engine [Ticker Bypass]")

# Core Calculation Engine
def apply_kalman(data, initial_p=100.0, q=0.0001, r=2.5):
    if len(data) == 0: return []
    x = data[0]; p = initial_p; filtered = []
    for z in data:
        p += q; k = p / (p + r); x += k * (z - x); p *= (1 - k)
        filtered.append(x)
    return np.array(filtered)

def apply_step_momentum(data):
    if len(data) == 0: return []
    x = data[0]; p = 1.0; q = 0.05; r = 0.2; filtered = []
    for z in data:
        p += q; k = p / (p + r); x += k * (z - x); p *= (1 - k)
        filtered.append(np.round(x))
    return filtered

# DATA PIPELINE (The Bypass)
@st.cache_data
def get_data_bypass():
    # Force Yahoo to recognize the ticker
    nifty = yf.Ticker("^NSEI")
    # Fetch data in chunks if needed or use history method
    df = nifty.history(period="2y", interval="1h")
    return df.ffill().dropna()

with st.spinner("Bypassing API restrictions..."):
    df = get_data_bypass()
    
    if len(df) > 100:
        # Core Calculations
        df['Kalman_Price'] = apply_kalman(df['Close'].values)
        df['Raw_WM'] = df['Close'] - df['Kalman_Price']
        df['Weighted_Momentum'] = apply_kalman(df['Raw_WM'].values, initial_p=0.5, q=0.01, r=0.5)
        df['Step_Momentum'] = apply_step_momentum(df['Weighted_Momentum'].values)
        
        # Accumulator
        df['Accumulator'] = 0
        score = 0
        for i in range(len(df)):
            if df['Weighted_Momentum'].iloc[i] > 0 and df['Step_Momentum'].iloc[i] > 0: score = min(score + 1, 5)
            elif df['Weighted_Momentum'].iloc[i] < 0 and df['Step_Momentum'].iloc[i] < 0: score = max(score - 1, -5)
            df.iloc[i, df.columns.get_loc('Accumulator')] = score

        # Split 50:50
        split_idx = int(len(df) * 0.50)
        display_df = df.iloc[split_idx:].sort_index(ascending=False)

        st.write(f"### Total Candles: {len(df)} | Live Window: {len(display_df)} Rows")
        st.dataframe(display_df[['Close', 'Weighted_Momentum', 'Step_Momentum', 'Accumulator']], use_container_width=True)
    else:
        st.error("Data fetch failed. ^NSEI 1h might be restricted. Try changing interval to '1d'.")
