import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty-VIX 90% Trap Filter 2026", layout="wide")
st.title("🎯 Nifty 50 + India VIX 1-Hour Traps & Divergence System")
st.write("Tracking Nifty (^NSEI) and VIX (^INDIAVIX) from January 1, 2026. Filtering the 90% Fake Out Moves.")

# Fetch 1-Hour Data for both Nifty 50 and India VIX concurrently
@st.cache_data(ttl=300)
def load_combined_data():
    # Downloading both tickers together to maintain exact timeline alignment
    df_raw = yf.download(tickers=["^NSEI", "^INDIAVIX"], start="2026-01-01", interval="1h")
    return df_raw

df_raw = load_combined_data()

if not df_raw.empty:
    # Processing Multi-index columns returned by yfinance for multiple tickers
    # Structure: (Metric, Ticker) -> We need to separate Nifty and VIX
    df = pd.DataFrame(index=df_raw.index)
    
    # Extracting Nifty Pieces
    df['N_High'] = df_raw[('High', '^NSEI')]
    df['N_Low'] = df_raw[('Low', '^NSEI')]
    df['N_Open'] = df_raw[('Open', '^NSEI')]
    df['N_Close'] = df_raw[('Close', '^NSEI')]
    
    # Extracting VIX Pieces
    df['V_High'] = df_raw[('High', '^INDIAVIX')]
    df['V_Low'] = df_raw[('Low', '^INDIAVIX')]
    df['V_Open'] = df_raw[('Open', '^INDIAVIX')]
    df['V_Close'] = df_raw[('Close', '^INDIAVIX')]
    
    df = df.reset_index()
    
    # Drop rows where data might be missing for either asset to keep exact row matching
    df = df.dropna().copy()
    
    # 1. Column D: Date and Time Formatter
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    # 2. Column A Calculations: (High + Low) / 2 for both assets
    df['Nifty_A'] = (df['N_High'] + df['N_Low']) / 2
    df['Vix_A'] = (df['V_High'] + df['V_Low']) / 2
    
    # 3. Column B Calculations: Exact Excel Drag-Down Loop Logic (Multiplier = 0.0001)
    multiplier = 0.0001
    n_col_b = np.zeros(len(df))
    v_col_b = np.zeros(len(df))
    
    if len(df) > 0:
        n_col_b[0] = df['Nifty_A'].iloc[0]  # First row Nifty = A1
        v_col_b[0] = df['Vix_A'].iloc[0]    # First row VIX = A1
        
    for i in range(1, len(df)):
        # Exact Excel cell trace replication
        n_col_b[i] = n_col_b[i-1] + (multiplier * (df['Nifty_A'].iloc[i] - n_col_b[i-1]))
        v_col_b[i] = v_col_b[i-1] + (multiplier * (df['Vix_A'].iloc[i] - v_col_b[i-1]))
        
    df['Nifty_B'] = n_col_b
    df['Vix_B'] = v_col_b
    
    # 4. Column C Calculations: A - B Deviation Profiler
    df['Nifty_C'] = df['Nifty_A'] - df['Nifty_B']
    df['Vix_C'] = df['Vix_A'] - df['Vix_B']
    
    # Extra Volatility Checks for Breakout verification
    df['Nifty_Range'] = df['N_High'] - df['N_Low']
    df['Avg_Nifty_Range'] = df['Nifty_Range'].rolling(window=20).mean()
    df['Nifty_Body'] = df['N_Close'] - df['N_Open']
    
    # 5. Column E: The 90% Opposite Secret Trick Logic Engine
    status_list = ["Baseline"]
    
    for i in range(1, len(df)):
        n_curr_c = df
