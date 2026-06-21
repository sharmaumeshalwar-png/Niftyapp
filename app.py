import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty DI Trap Filter Stable", layout="wide")
st.title("🎯 Nifty 50 Pure Math +DI / -DI 1-Hour Trap Filter System")
st.write("Tracking Nifty 50 (^NSEI) from June 1, 2026. Anti-crash index mechanism deployed.")

# Fetch 1-Hour Accurate Nifty 50 Data safely
@st.cache_data(ttl=300)
def load_pure_data():
    df_raw = yf.download(tickers="^NSEI", start="2026-06-01", interval="1h")
    if df_raw.empty:
        return pd.DataFrame()
        
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
        
    # Drop completely empty or mismatched rows to avoid script execution error
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
    # 🛠️ SAFE MATHEMATICAL +DI / -DI ENGINE CALCULATION (14 Period)
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
    
    plus_dm = np.zeros(len(df))
    minus_dm = np.zeros(len(df))
    
    for i in range(1, len(df)):
        up = df['UpMove'].iloc[i]
        down = df['DownMove'].iloc[i]
        
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
                
    df['+DM'] = plus_dm
    df['-DM'] = minus_dm
    
    # Safe Wilder's Smoothing
    tr_smooth = np.zeros(len(df))
    p_dm_smooth = np.zeros(len(df))
    m_dm_smooth = np.zeros(len(df))
    
    if len(df) >= 15:
        # Initial sum for base anchoring
        tr_smooth[14] = df['TR'].iloc[1:15].sum()
        p_dm_smooth[14] = df['+DM'].iloc[1:15].sum()
        m_dm_smooth[14] = df['-DM'].iloc[1:15].sum()
        
        for i in range(15, len(df)):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / 14) + df['TR'].iloc[i]
            p_dm_smooth[i] = p_dm_smooth[i-1] - (p_dm_smooth[i-1] / 14) + df['+DM'].iloc[i]
            m_dm_smooth[i] = m_dm_smooth[i-1] - (m_dm_smooth[i-1] / 14) + df['-DM'].iloc[i]
            
    df['TR_Smooth'] = tr_smooth
    df['+DM_Smooth'] = p_dm_smooth
    df['-DM_Smooth'] = m_dm_smooth
    
    # Final DI Percentages with zero division handling
    df['+DI'] = np.where(df['TR_Smooth'] > 0, (df['+DM_Smooth'] / df['TR_Smooth']) * 100, 0)
    df['-DI'] = np.where(df['TR_Smooth'] > 0, (df['-DM_Smooth'] / df['TR_Smooth']) * 100, 0)
    
    # ---------------------------------------------------------
    # 🎯 COLUMN E: CROSS-VALIDATION INTERSECTING ENGINE (DI Traps)
    # ---------------------------------------------------------
    status_list = []
    
    for i in range(len(df)):
        if i < 15:
            status_list.append("💤 INITIALIZING SYSTEM")
            continue
            
        curr_c = df['Column C'].iloc[i]
        p_di = df['+DI'].iloc[i]
        m_di = df['-DI'].iloc[i]
        
        if curr_c > 0:  # Nifty Plus (+) Zone
            if p_di > m_di:
                status_list.append("🟢 TRUE BULLISH MOMENTUM (+DI Confirmed)")
            else:
                status_list.append("⚠️ 90% FAKE UPMOVE TRAP (-DI Dominating!)")
        else:  # Nifty Minus (-) Zone
            if m_di > p_di:
                status_list.append("🔴 TRUE BEARISH CRASH (-DI Confirmed)")
            else:
                status_list.append("⚠️ 90% FAKE DOWNMOVE TRAP (+DI Dominating!)")
                
    df['Column E'] = status_list
    
    # Clean baseline visualization filter
    df = df[df['Raw_Date'] >= '2026-06-01'].copy()
    
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', '+DI', '-DI', 'Column E']].copy()
    show_df = show_df.iloc[::-1]
    
    def color_trap_grid(val):
        if "🟢" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "🔴" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "⚠️" in val: return 'background-color: #d35400; color: white; font-weight: bold;'
        if "💤" in val: return 'background-color: #34495e; color: #bdc3c7;'
        return ''

    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
        '+DI': '{:.2f}', '-DI': '{:.2f}'
    }).map(color_trap_grid, subset=['Column E']), use_container_width=True)
else:
    st.error("Market data pipeline break ho gaya hai.")
