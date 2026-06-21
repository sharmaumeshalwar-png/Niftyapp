import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty Pure VIX System", layout="wide")
st.title("🎯 Nifty 50 + India VIX Direct Candle Cross-System")
st.write("Tracking Nifty (^NSEI) with exact Excel formula and cross-checking directly with India VIX (^INDIAVIX) movement from June 1, 2026.")

@st.cache_data(ttl=300)
def load_pure_data():
    df_nifty = yf.download(tickers="^NSEI", start="2026-06-01", interval="1d")
    df_vix = yf.download(tickers="^INDIAVIX", start="2026-06-01", interval="1d")
    
    if isinstance(df_nifty.columns, pd.MultiIndex):
        df_nifty.columns = df_nifty.columns.get_level_values(0)
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.get_level_values(0)
        
    if df_nifty.empty or df_vix.empty:
        return pd.DataFrame()
        
    nifty_clean = pd.DataFrame(index=df_nifty.index)
    nifty_clean['N_Open'] = df_nifty['Open']
    nifty_clean['N_High'] = df_nifty['High']
    nifty_clean['N_Low'] = df_nifty['Low']
    nifty_clean['N_Close'] = df_nifty['Close']
    
    vix_clean = pd.DataFrame(index=df_vix.index)
    vix_clean['V_Close'] = df_vix['Close']  # Direct daily closing price for VIX
    
    return pd.merge(nifty_clean, vix_clean, left_index=True, right_index=True, how='inner')

df = load_pure_data()

if not df.empty:
    df = df.reset_index()
    
    time_col = 'Date' if 'Date' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y')
    
    # 1. Column A: Nifty (High + Low) / 2
    df['Column A'] = (df['N_High'] + df['N_Low']) / 2
    
    # 2. Column B: Exact Excel Drag-Down Loop Formula for Nifty
    multiplier = 0.0001
    n_col_b = np.zeros(len(df))
    
    if len(df) > 0:
        n_col_b[0] = df['Column A'].iloc[0]  # First row = A1
        
    for i in range(1, len(df)):
        n_col_b[i] = n_col_b[i-1] + (multiplier * (df['Column A'].iloc[i] - n_col_b[i-1]))
        
    df['Column B'] = n_col_b
    
    # 3. Column C: Exact Formula -> A - B
    df['Column C'] = df['Column A'] - df['Column B']
    
    # 4. India VIX Direct Direction (Current Candle vs Previous Candle)
    df['VIX_Prev_Close'] = df['V_Close'].shift(1)
    
    # 5. Column E: Pure Cross-Check Hint Engine
    status_list = ["Baseline"]
    
    for i in range(1, len(df)):
        n_curr_c = df['Column C'].iloc[i]
        v_close = df['V_Close'].iloc[i]
        v_prev = df['VIX_Prev_Close'].iloc[i]
        
        # Checking if VIX is strictly rising or falling
        vix_is_up = v_close > v_prev
        
        # --- DIRECT INTERSECTION LOGIC ---
        if n_curr_c > 0:  # Nifty is Positive (+)
            if not vix_is_up:  # VIX is Falling (Normal Market)
                status_list.append("🟢 TRUE BULLISH BREAKOUT (VIX Down)")
            else:  # VIX is Rising (Divergence Trap)
                status_list.append("⚠️ 90% FAKE UPMOVE TRAP (VIX Rising!)")
                
        else:  # Nifty is Negative (-)
            if vix_is_up:  # VIX is Rising (Normal Panic)
                status_list.append("🔴 TRUE BEARISH CRASH (VIX Up)")
            else:  # VIX is Falling (Divergence Trap)
                status_list.append("⚠️ 90% FAKE DOWNMOVE TRAP (VIX Dropping!)")
                
    df['Column E'] = status_list
    
    # Filter to strictly show from June 1, 2026 onwards
    df = df[df['Raw_Date'] >= '2026-06-01'].copy()
    
    # Reverse layout to show the latest candle on top row
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'V_Close', 'Column E']].rename(
        columns={'V_Close': 'India_VIX'}
    ).copy()
    show_df = show_df.iloc[::-1]
    
    # Custom Visual Grid Theme Engine
    def color_trap_grid(val):
        if "🟢" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "🔴" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "⚠️" in val: return 'background-color: #d35400; color: white; font-weight: bold;'
        return ''

    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 'India_VIX': '{:.2f}'
    }).map(color_trap_grid, subset=['Column E']), use_container_width=True)
else:
    st.error("Data Load Error.")
