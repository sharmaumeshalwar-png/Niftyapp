import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty +DI/-DI Trap Filter 2026", layout="wide")
st.title("🎯 Nifty 50 Pure Math +DI / -DI 1-Hour Trap Filter System")
st.write("Tracking Nifty 50 (^NSEI) from June 1, 2026. Filtering the 90% Fake Out Moves using Directional Indicators.")

# Fetch 1-Hour Accurate Nifty 50 Data from June 1, 2026
@st.cache_data(ttl=300)
def load_pure_data():
    df_raw = yf.download(tickers="^NSEI", start="2026-06-01", interval="1h")
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
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
    
    # 3. Column B: Exact Excel Drag-Down Loop Logic (Multiplier = 0.0001)
    multiplier = 0.0001
    n_col_b = np.zeros(len(df))
    
    if len(df) > 0:
        n_col_b[0] = df['Column A'].iloc[0]  # First row = A1 logic
        
    for i in range(1, len(df)):
        n_col_b[i] = n_col_b[i-1] + (multiplier * (df['Column A'].iloc[i] - n_col_b[i-1]))
        
    df['Column B'] = n_col_b
    
    # 4. Column C: Exact Formula -> A - B Deviation
    df['Column C'] = df['Column A'] - df['Column B']
    
    # ---------------------------------------------------------
    # 🛠️ PURE MATHEMATICAL +DI / -DI ENGINE CALCULATION (14 Period)
    # ---------------------------------------------------------
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Close'] = df['Close'].shift(1)
    
    # Calculate True Range (TR)
    df['TR1'] = df['High'] - df['Low']
    df['TR2'] = (df['High'] - df['Prev_Close']).abs()
    df['TR3'] = (df['Low'] - df['Prev_Close']).abs()
    df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    
    # Calculate Directional Movements (+DM and -DM)
    df['UpMove'] = df['High'] - df['Prev_High']
    df['DownMove'] = df['Prev_Low'] - df['Low']
    
    plus_dm = []
    minus_dm = []
    
    for i in range(len(df)):
        if i == 0:
            plus_dm.append(0)
            minus_dm.append(0)
        else:
            up = df['UpMove'].iloc[i]
            down = df['DownMove'].iloc[i]
            
            if up > down and up > 0:
                plus_dm.append(up)
            else:
                plus_dm.append(0)
                
            if
