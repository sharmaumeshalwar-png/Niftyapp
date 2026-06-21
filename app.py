import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty Micro-Hull Structural System", layout="wide")
st.title("🎯 Nifty 50 Dual-Stage Micro-Structural Velocity Engine")
st.write("De-trending Nifty 50 (^NSEI) from May 1, 2026. Catching 90% Traps via Pure Iterative Memory Gaps.")

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
    # 🛠️ PURE MATHEMATICAL MICRO-LEVEL TRACKING SEQUENCES
    # ---------------------------------------------------------
    # Sequence B1 (Fast Track Multiplier = 0.01)
    m_fast = 0.01
    b1_seq = np.zeros(len(df))
    
    # Sequence B2 (Slow Track Multiplier = 0.0001)
    m_slow = 0.0001
    b2_seq = np.zeros(len(df))
    
    if len(df) > 0:
        b1_seq[0] = df['Column A'].iloc[0]
        b2_seq[0] = df['Column A'].iloc[0]
        
    for i in range(1, len(df)):
        b1_seq[i] = b1_seq[i-1] + (m_fast * (df['Column A'].iloc[i] - b1_seq[i-1]))
        b2_seq[i] = b2_seq[i-1] + (m_slow * (df['Column A'].iloc[i] - b2_seq[i-1]))
        
    df['B1_Fast'] = b1_seq
    df['B2_Slow'] = b2_seq
    
    # Calculate the Structural Micro Gap
    df['Micro_Gap'] = df['B1_Fast'] - df['B2_Slow']
    
    # Sequence B3 (Catalyst Tracker on the Micro Gap itself - Multiplier = 0.05)
    m_catalyst = 0.05
    b3_seq = np.zeros(len(df))
    
    if len(df) > 0:
        b3_seq[0] = df['Micro_Gap'].iloc[0]
        
    for i in range(1, len(df)):
        b3_seq[i] = b3_seq[i-1] + (m_catalyst * (df['Micro_Gap'].iloc[i] - b3_seq[i-1]))
        
    df['B3_Catalyst'] = b3_seq
