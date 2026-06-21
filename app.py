import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty Price-Density Structural System", layout="wide")
st.title("🎯 Nifty 50 Mathematical Price-Density Variance Engine")
st.write("De-trending Nifty 50 (^NSEI) from May 1, 2026. Filtering 90% Traps via Micro-Space Compression.")

# Fetch 1-Hour Accurate Nifty 50 Data safely
@st.cache_data(ttl=300)
def load_pure_data():
    df_raw = yf.download(tickers="^NSEI", start="2026-05-01", interval="1h")
    if df_raw.empty:
        return pd.DataFrame()
        
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
        
    # Drop empty or corrupted data rows instantly
    df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    return df_raw

df = load_pure_data()

if not df.empty:
    df = df.reset_index()
    
    # 1. Column D: Date and Time Formatter
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    # 2. Column A: Exact Formula -> (High + Low) / 2
    df['Column A'] = (df['High'] + df['Low']) / 2
    
    # 3. Column B: Your Original Base Loop (Multiplier = 0.0001)
    multiplier_base = 0.0001
    n_col_b = np.zeros(len(df))
    
    if len(df) > 0:
        n_col_b[0] = df['Column A'].iloc[0]
        
    for i in range(1, len(df)):
        n_col_b[i] = n_col_b[i-1] + (multiplier_base * (df['Column A'].iloc[i] - n_col_b[i-1]))
        
    df['Column B'] = n_col_b
    
    # 4. Column C: Exact Formula -> A - B Deviation
    df['Column C'] = df['Column A'] - df['Column B']
    
    # ---------------------------------------------------------
    # 🛠️ PURE MATHEMATICAL MICRO-DENSITY TRACKING ARRAYS
    # ---------------------------------------------------------
    # Micro-Range Dispersion Track
    df['Candle_Range'] = df['High'] - df['Low']
    
    # Running Average Loop for Volatility Base (Multiplier = 0.01 for Micro-Breathing Space)
    multiplier_range = 0.01
    range_b = np.zeros(len(df))
