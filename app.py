import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Page Configuration Setup
st.set_page_config(page_title="Nifty E-J Cascade Fixed", layout="wide")
st.title("🎯 Nifty 50 5-Stage Pure Sign Cascade System (E - J)")
st.write("Column K is strictly calculated as: (Sign of E) - (Sign of J). Displaying data from January 1, 2026.")

# Robust Data Fetcher with Fallback to prevent blank screen
@st.cache_data(ttl=300)
def load_pure_data():
    try:
        # Attempt 1: Fetch from Jan 1, 2026
        df_raw = yf.download(tickers="^NSEI", start="2026-01-01", interval="1h")
        
        # Fallback Attempt 2: If empty, pull trailing data to break the API freeze
        if df_raw.empty or len(df_raw) < 10:
            df_raw = yf.download(tickers="^NSEI", period="730d", interval="1h")
            
        if df_raw.empty:
            return pd.DataFrame()
            
        # Flatten MultiIndex columns if present
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
            
        # Remove empty rows safely
        df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
        return df_raw
    except Exception as e:
        st.error(f"API Fetch Error: {str(e)}")
        return pd.DataFrame()

df = load_pure_data()

if not df.empty:
    df = df.reset_index()
    
    # Column D: Date and Time Formatter Block
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
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
        col_e[i] = col_e[i-1] + (multiplier * (float(df['Column C'].values[i]) - col_e[i-1]))
    df['Column E'] = col_e
    
    # 5. Column F: Next-Gen Deviation -> C - E
    df['Column F'] = (df['Column C'] - df['Column E']).astype(float)
    
    # 6. Column G: Smooth Loop of Column F (Stage 3 Stable)
    col_g = np.zeros(total_rows, dtype=float)
    if total_rows > 0:
        col_g[0] = float(df['Column F'].values[0])
    for i in range(1, total_rows):
        col_g[i] = col_g[i-1] + (multiplier * (float(df['Column F'].values[i]) - col_g[i-1]))
    df['Column G'] = col_g
    
    # 7. Column H: Third-Gen Deviation -> F - G
    df['Column H'] = (df['Column F'] - df['Column G']).astype(float)
    
    # 8. Column I: Smooth Loop of Column H (Stage 4 Stable)
    col_i = np.zeros(total_rows, dtype=float)
    if total_rows > 0:
        col_i[0] = float(df['Column H'].values[0])
    for i in range(1, total_rows):
        col_i[i] = col_i[i-1] + (multiplier * (float(df['Column H'].values[i]) - col_i[i-1]))
    df['Column I'] = col_i
    
    # 9. Column J: Smooth Loop of Column I (Stage 5 Stable)
    col_j = np.zeros(total_rows, dtype=float)
    if total_rows > 0:
        col_j[0] = float(col_i[0])
    for i in range(1, total_rows):
        col_j[i] = col_j[i-1] + (multiplier * (col_i[i] - col_j[i-1]))
    df['Column J'] = col_j
    
    # 🛠️ 10. COLUMN K: SIGN-SUBTRACTION CORE -> (Sign of E) - (Sign of J)
    sign_e = np.sign(col_e).astype(float)
    sign_j = np.sign(col_j).astype(float)
    df['Column K'] = sign_e - sign_j
    
    # 🌟 SIGN-DIFFERENCE LOOP VERIFICATION ENGINE (E - J Match)
    status_list = ["System Booting"]
    for i in range(1, total_rows):
        curr_k = float(df['Column K'].values[i])
        curr_c = float(df['Column C'].values[i])
        
        if curr_c > 0:  # Surface price is showing Plus (+)
            if curr_k == 2.0 or curr_k == 0.0:  
                status_list.append("🟢 SIGN BULLISH (E-J Confirmed)")
            else:  
                status_list.append("⚠️ 90% CALL TRAP (E-J Inversion Alert!)")
        else:  # Surface price is showing Minus (-)
            if curr_k == -2.0 or curr_k == 0.0:  
                status_list.append("🔴 SIGN BEARISH (E-J Confirmed)")
            else:  
                status_list.append("⚠️ 90% PUT TRAP (E-J Inversion Alert!)")
                
    df['Signal_Status'] = status_list

    # 🔥 STABLE DISPLAY FILTER: Purana historical data peeche calculate hoga, par UI par strictly 1 Jan 2026 se dikhega
    df_filtered = df[df['Raw_Date'] >= '2026-01-01'].copy()
    
    if df_filtered.empty:
        df_filtered = df.copy() # Safe
