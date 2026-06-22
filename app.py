import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty E-J Fixed 2026", layout="wide")
st.title("🎯 Nifty 50 5-Stage Pure Sign Cascade System (E - J)")
st.write("Strictly Displaying Data from January 1, 2026. Powered by Pure Sign-Difference Matrix (E - J).")

# Fetch Data safely from Jan 1st, 2026
@st.cache_data(ttl=300)
def load_pure_data():
    # Fetch data cleanly
    df_raw = yf.download(tickers="^NSEI", start="2026-01-01", interval="1h")
    
    if df_raw.empty:
        return pd.DataFrame()
        
    # Flatten MultiIndex columns if yfinance returns them
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
        
    # Clean empty rows
    df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    return df_raw

df = load_pure_data()

if not df.empty:
    df = df.reset_index()
    
    # Column D: Date and Time Formatter with STRICT Timezone Removal (Fixes Blank Screen)
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col]).dt.tz_localize(None) # 🔥 CRITICAL FIX
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    total_rows = len(df)
    multiplier = 0.0001
    
    # 1. Column A: Exact Formula -> (High + Low) / 2
    df['Column A'] = ((df['High'] + df['Low']) / 2.0).astype(float)
    
    # 2. Column B: Smooth Loop of Column A (Stage 1 Stable)
    col_b = np.zeros(total_rows, dtype=float)
    if total_rows > 0:
        col_b[0] = float(df['Column A'].values[0])
    for i in range(1, total_rows):
        col_b[i] = col_b[i-1] + (multiplier * (float(df['Column A'].values[i]) - col_b[i-1]))
    df['Column B'] = col_b
    
    # 3. Column C: Pure Deviation -> A - B
    df['Column C'] = (df['Column A'] - df['Column B']).astype(float)
    
    # 4. Column E: Smooth Loop of Column C (Stage 2 Stable)
    col_e = np.zeros(total_rows, dtype=float)
    if total_rows > 0:
        col_e[0] = float(df['Column C'].values[0])
    for i in range(1, total_rows):
        col_e[i] = col_e[i-1]
