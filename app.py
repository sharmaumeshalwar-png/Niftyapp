import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty Cascade Chain March", layout="wide")
st.title("🎯 Nifty 50 Infinite Cascading Loop System")
st.write("Multi-Stage Price Stabilization Chain running from March 1, 2026. Every step filters the previous deviation.")

# Fetch 1-Hour Accurate Nifty 50 Data safely from 1st March
@st.cache_data(ttl=300)
def load_pure_data():
    df_raw = yf.download(tickers="^NSEI", start="2026-03-01", interval="1h")
    if df_raw.empty:
        return pd.DataFrame()
        
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
        
    df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    return df_raw

df = load_pure_data()

if not df.empty:
    df = df.reset_index()
    
    # Column D: Date and Time Formatter
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    # Total rows available for the sequential loop
    total_rows = len(df)
    multiplier = 0.0001
    
    # 1. Column A: Exact Formula -> (High + Low) / 2
    df['Column A'] = (df['High'] + df['Low']) / 2.0
    
    # 2. Column B: Smooth Loop of Column A
    col_b = np.zeros(total_rows)
    if total_rows > 0:
        col_b[0] = float(df['Column A'].iloc[0])
    for i in range(1, total_rows):
        col_b[i] = col_b[i-1] + (multiplier * (float(df['Column A'].iloc[i]) - col_b[i-1]))
    df['Column B'] = col_b
    
    # 3. Column C: Pure Deviation -> A - B
    df['Column C'] = df['Column A'] - df['Column B']
    
    # 4. Column E: Smooth Loop of Column C (Stabilizing C)
    col_e = np.zeros(total_rows)
    if total_rows > 0:
        col_e[0] = float(df['Column C'].iloc[0])
    for i in range(1, total_rows):
        col_e[i] = col_e[i-1] + (multiplier * (float(df['Column C'].iloc[i]) - col_e[i-1]))
    df['Column E'] = col_e
    
    # 5. Column F: Next-Gen Deviation -> C - E
    df['Column F'] = df['Column C'] - df['Column E']
    
    # 6. Column G: Smooth Loop of Column F (Stabilizing F)
    col_g = np.zeros(total_rows)
    if total_rows > 0:
        col_g[0] = float(df['Column F'].iloc[0])
    for i in range(1, total_rows):
        col_g[i] = col_g[i-1] + (multiplier * (float(df['Column F'].iloc[i]) - col_g[i-1]))
    df['Column G'] = col_g
    
    # 🌟 NEW SUPER SIGNAL ENGINE (Based on Final G Column Velocity)
    status_list = ["System Booting"]
    for i in range(1, total_rows):
        curr_g = col_g[i]
        prev_g = col_g[i-1]
        curr_c = df['Column C'].iloc[i]
        
        g_is_rising = curr_g > prev_g
        
        if curr_c > 0:  # Surface price is
