import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty 4-Stage Cascade Jan 1", layout="wide")
st.title("🎯 Nifty 50 4-Stage Infinite Cascading Loop System")
st.write("Deep Mathematical Price Stabilization Chain running and displaying continuously from January 1, 2026.")

# Fetch Data safely from Jan 1st
@st.cache_data(ttl=300)
def load_pure_data():
    df_raw = yf.download(tickers="^NSEI", start="2026-01-01", interval="1h")
    
    if df_raw.empty:
        return pd.DataFrame()
        
    # Strictly flattening columns if yfinance returns MultiIndex headers
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
        
    # Clean rows with missing price marks to stop loop breakdown
    df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    return df_raw

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
    
    # 🌟 ULTIMATE 4-STAGE SIGNAL ENGINE (Based on Final Column I Velocity)
    status_list = ["System Booting"]
    for i in range(1, total_rows):
        curr_i = col_i[i]
        prev_i = col_i[i-1]
        curr_c = float(df['Column C'].values[i])
        
        i_is_rising = curr_i > prev_i
        
        if curr_c > 0:  # Surface price is showing Plus (+)
            if i_is_rising:  # Deep 4th layer sequence is expanding up -> 10% Real Move
                status_list.append("🟢 4-STAGE BULLISH (High Conviction)")
            else:  # Surface is plus but deep micro-stabilizer is falling -> 90% CALL TRAP
                status_list.append("⚠️ 90% CALL TRAP (Deep Cascade Alert!)")
        else:  # Surface price is showing New/Minus (-)
            if not i_is_rising:  # Deep 4th layer sequence is breaking down -> Real Crash
                status_list.append("🔴 4-STAGE BEARISH (High Conviction)")
            else:  # Surface is minus but deep micro-stabilizer is absorbing -> 90% PUT TRAP
                status_list.append("⚠️ 90% PUT TRAP (Short Covering!)")
                
    df['Signal_Status'] = status_list

    # Pure Raw Render without any date trimming - strictly from Jan 1st to latest candle
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column F', 'Column G', 'Column H', 'Column I', 'Signal_Status']].copy()
    show_df = show_df.iloc[::-1]  # Latest candle on top
    
    # Grid Theme Color Engine
    def color_trap_grid(val):
        if "🟢" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "🔴" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "⚠️" in val: return 'background-color: #d35400; color: white; font-weight: bold;'
        return ''

    # Dynamic Frame Render with explicit floating formats up to 4 decimal precision
    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
        'Column E': '{:.4f}', 'Column F': '{:.4f}', 'Column G': '{:.4f}',
        'Column H': '{:.4f}', 'Column I': '{:.4f}'
    }).map(color_trap_grid, subset=['Signal_Status']), use_container_width=True)
else:
    st.error("Data pipeline load error: Data stream structure couldn't be mounted on Streamlit.")
