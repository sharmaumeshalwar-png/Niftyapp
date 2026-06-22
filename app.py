import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="NiftyBees Institutional Guard", layout="wide")
st.title("🎯 Nifty Bees (ETF) 5-Stage Multi-Indicator Cascade")
st.write("Column K = (Sign of E) - (Sign of I). Tracking physical volumes and RSI on NIFTYBEES.NS from Jan 1, 2025.")

# Fetch Data safely for Nifty Bees
@st.cache_data(ttl=300)
def load_pure_data():
    # NIFTYBEES.NS ticker use kar rahe hain real traded volume ke liye
    df_raw = yf.download(tickers="NIFTYBEES.NS", start="2025-01-01", interval="1h")
    if df_raw.empty:
        return pd.DataFrame()
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume']).copy()
    return df_raw

df = load_pure_data()

if not df.empty:
    df = df.reset_index()
    
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    total_rows = len(df)
    multiplier = 0.0001
    
    # --- 1. CORE PRICE CASCADE BLOCKS ---
    df['Column A'] = ((df['High'] + df['Low']) / 2.0).astype(float)
    
    col_b = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_b[0] = float(df['Column A'].values[0])
    for i in range(1, total_rows):
        col_b[i] = col_b[i-1] + (multiplier * (float(df['Column A'].values[i]) - col_b[i-1]))
    df['Column B'] = col_b
    
    df['Column C'] = (df['Column A'] - df['Column B']).astype(float)
    
    col_e = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_e[0] = float(df['Column C'].values[0])
    for i in range(1, total_rows):
        col_e[i] = col_e[i-1] + (multiplier * (float(df['Column C'].values[i]) - col_e[i-1]))
    df['Column E'] = col_e
    
    df['Column F'] = (df['Column C'] - df['Column E']).astype(float)
    
    col_g = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_g[0] = float(df['Column F'].values[0])
    for i in range(1, total_rows):
        col_g[i] = col_g[i-1] + (multiplier * (float(df['Column F'].values[i]) - col_g[i-1]))
    df['Column G'] = col_g
    
    df['Column H'] = (df['Column F'] - df['Column G']).astype(float)
    
    col_i = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_i[0] = float(df['Column H'].values[0])
    for i in range(1, total_rows):
        col_i[i] = col_i[i-1] + (multiplier * (float(df['Column H'].values[i]) - col_i[i-1]))
    df['Column I'] = col_i
    
    col_j = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_j[0] = float(col_i[0])
    for i in range(1, total_rows):
        col_j[i] = col_j[i-1] + (multiplier * (col_i[i] - col_j[i-1]))
    df['Column J'] = col_j
    
    # Master Sign Block (Sign of E - Sign of I)
    sign_e = np.sign(col_e).astype(float)
    sign_i = np.sign(col_i).astype(float)
    df['Column K'] = sign_e - sign_i
    
    # --- 2. REAL VOLUME & MOMENTUM ENGINES ---
    # Real Traded Volume Moving Average (20 hours baseline)
    df['Vol_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    
    # RSI 14 Math Block
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # --- 3. SIGNAL ENGINE WITH OPERATOR GUARD ---
    status_list = ["System Booting"]
    for i in range(1, total_rows):
        curr_k = float(df['Column K'].values[i])
        curr_c = float(df['Column C'].values[i])
        curr_vol = float(df['Volume'].values[i])
        avg_vol = float(df['Vol_MA'].values[i])
        curr_rsi = float(df['RSI'].values[i])
        
        # Real Traded Volume Spike Filter
        volume_is_huge = curr_vol > (avg_vol * 2.2)  # Volume is 2.2x normal hourly flow
        rsi_overbought = curr_rsi > 70.0
        rsi_oversold = curr_rsi < 30.0
        
        if curr_c > 0:  
            if curr_k == 2.0 or curr_k == 0.0:
                if volume_is_huge and rsi_overbought:
                    status_list.append("🚨 NIFTYBEES DUMP (Call Squeeze Danger!)")
                else:
                    status_list.append("🟢 SIGN BULLISH (E-I Confirmed)")
            else:
                status_list.append("⚠️ 90% CALL TRAP (E-I Inversion Alert!)")
                
        else:  
            if curr_k == -2.0 or curr_k == 0.0:
                if volume_is_huge and rsi_oversold:
                    status_list.append("🚨 NIFTYBEES ABSORPTION (Put Squeeze Danger!)")
                else:
                    status_list.append("🔴 SIGN BEARISH (E-I Confirmed)")
            else:
                status_list.append("⚠️ 90% PUT TRAP (E-I Inversion Alert!)")
                
    df['Signal_Status'] = status_list

    # Presentation Layer Link (Latest candle on top)
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column I', 'Column K', 'Volume', 'RSI', 'Signal_Status']].copy()
    show_df = show_df.iloc[::-1]  
    
    # Grid Theme Color Engine
    def color_trap_grid(val):
        if "🟢" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "🔴" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "⚠️" in val: return 'background-color: #d35400; color: white; font-weight: bold;'
        if "🚨" in val: return 'background-color: #7b1113; color: #ffffff; font-weight: bold; border: 2px solid yellow;'
        return ''

    # Render Frame safely with real Volume column shown
    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
        'Column E': '{:.4f}', 'Column I': '{:.4f}', 'RSI': '{:.2f}',
        'Volume': '{:,.0f}', 'Column K': '{:.0f}'
    }).map(color_trap_grid, subset=['Signal_Status']), use_container_width=True)
else:
    st.error("Data pipeline load error: NIFTYBEES.NS data stream unavailable.")
