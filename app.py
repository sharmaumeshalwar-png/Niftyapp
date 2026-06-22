import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty VIX-Guarded Cascade", layout="wide")
st.title("🎯 Nifty 50 VIX-Guarded 5-Stage Cascade System")
st.write("Column K = (Sign of E) - (Sign of I). Real-time risk mitigation powered by India VIX Correlation Engine.")

# Fetch both Nifty Spot and India VIX safely from Jan 1, 2025
@st.cache_data(ttl=300)
def load_synchronized_data():
    # Fetching Nifty Spot for high-precision price and India VIX for structural fear tracking
    df_nifty = yf.download(tickers="^NSEI", start="2025-01-01", interval="1h")
    df_vix = yf.download(tickers="^INDIAVIX", start="2025-01-01", interval="1h")
    
    if df_nifty.empty or df_vix.empty:
        return pd.DataFrame()
        
    if isinstance(df_nifty.columns, pd.MultiIndex):
        df_nifty.columns = df_nifty.columns.get_level_values(0)
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.get_level_values(0)
        
    df_nifty = df_nifty.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    
    # Map VIX Close directly to Nifty's timeline
    df_nifty['VIX_Close'] = df_vix['Close']
    df_nifty['VIX_Close'] = df_nifty['VIX_Close'].ffill().bfill()
    return df_nifty

df = load_synchronized_data()

if not df.empty:
    df = df.reset_index()
    
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    total_rows = len(df)
    multiplier = 0.0001
    
    # --- 1. CORE MATH CASCADE BLOCKS ---
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
    
    # Master Sign Block
    sign_e = np.sign(col_e).astype(float)
    sign_i = np.sign(col_i).astype(float)
    df['Column K'] = sign_e - sign_i
    
    # --- 2. VIX VELOCITY FILTER ENGINE ---
    # Tracking a 5-hour momentum window of India VIX
    df['VIX_Change'] = df['VIX_Close'].diff(periods=2)
    
    # --- 3. CONVERGENCE SIGNAL GENERATION ---
    status_list = ["System Booting"]
    for i in range(1, total_rows):
        curr_k = float(df['Column K'].values[i])
        curr_c = float(df['Column C'].values[i])
        vix_move = float(df['VIX_Change'].values[i])
        
        # Divergence Conditions: VIX spiking during rally or dropping during crash abnormal behavior
        vix_is_spiking = vix_move > 0.45  # Volatility sharply rising
        vix_is_crashing = vix_move < -0.45  # Volatility sharply dropping
        
        if curr_c > 0:  # Surface price shows Positive (+)
            if curr_k == 2.0 or curr_k == 0.0:
                if vix_is_spiking:
                    status_list.append("🚨 VIX ANOMALY DIVERGÈNCE (Huge Counter-Move Imminent!)")
                else:
                    status_list.append("🟢 SIGN BULLISH (E-I Confirmed)")
            else:
                status_list.append("⚠️ 90% CALL TRAP (E-I Inversion Alert!)")
                
        else:  # Surface price shows Negative (-)
            if curr_k == -2.0 or curr_k == 0.0:
                if vix_is_crashing:
                    status_list.append("🚨 VIX ANOMALY REVERSAL (Sudden Short Squeeze Imminent!)")
                else:
                    status_list.append("🔴 SIGN BEARISH (E-I Confirmed)")
            else:
                status_list.append("⚠️ 90% PUT TRAP (E-I Inversion Alert!)")
                
    df['Signal_Status'] = status_list

    # Presentation Output Rendering
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column I', 'Column K', 'VIX_Close', 'Signal_Status']].copy()
    show_df = show_df.iloc[::-1]  # Keep latest candle on top
    
    def color_trap_grid(val):
        if "🟢" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "🔴" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "⚠️" in val: return 'background-color: #d35400; color: white; font-weight: bold;'
        if "🚨" in val: return 'background-color: #7b1113; color: #ffffff; font-weight: bold; border: 2px solid yellow;'
        return ''

    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
        'Column E': '{:.4f}', 'Column I': '{:.4f}', 'VIX_Close': '{:.2f}',
        'Column K': '{:.0f}'
    }).map(color_trap_grid, subset=['Signal_Status']), use_container_width=True)
else:
    st.error("Data pipeline load error: Structural Index stream failed.")
