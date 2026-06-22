import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty Cascade May Filtered", layout="wide")
st.title("🎯 Nifty 50 Infinite Cascading Loop System")
st.write("Multi-Stage Price Stabilization Chain Displaying Data from May 1, 2026 (Backend Data Anchored from Jan 1 for High Precision).")

# Fetch Data safely from Jan 1st to give enough historical depth for 0.0001 multiplier
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
    
    # 2. Column B: Smooth Loop of Column A (B2 track stable)
    col_b = np.zeros(total_rows, dtype=float)
    if total_rows > 0:
        col_b[0] = float(df['Column A'].values[0])
    for i in range(1, total_rows):
        col_b[i] = col_b[i-1] + (multiplier * (float(df['Column A'].values[i]) - col_b[i-1]))
    df['Column B'] = col_b
    
    # 3. Column C: Pure Deviation -> A - B (Stable C)
    df['Column C'] = (df['Column A'] - df['Column B']).astype(float)
    
    # 4. Column E: Smooth Loop of Column C (E3 track stable)
    col_e = np.zeros(total_rows, dtype=float)
    if total_rows > 0:
        col_e[0] = float(df['Column C'].values[0])
    for i in range(1, total_rows):
        col_e[i] = col_e[i-1] + (multiplier * (float(df['Column C'].values[i]) - col_e[i-1]))
    df['Column E'] = col_e
    
    # 5. Column F: Next-Gen Deviation -> C - E (F4 track stable)
    df['Column F'] = (df['Column C'] - df['Column E']).astype(float)
    
    # 6. Column G: Smooth Loop of Column F (G5 track stable)
    col_g = np.zeros(total_rows, dtype=float)
    if total_rows > 0:
        col_g[0] = float(df['Column F'].values[0])
    for i in range(1, total_rows):
        col_g[i] = col_g[i-1] + (multiplier * (float(df['Column F'].values[i]) - col_g[i-1]))
    df['Column G'] = col_g
    
    # 🌟 NEW SUPER SIGNAL ENGINE (Based on Final G Column Velocity)
    status_list = ["System Booting"]
    for i in range(1, total_rows):
        curr_g = col_g[i]
        prev_g = col_g[i-1]
        curr_c = float(df['Column C'].values[i])
        
        g_is_rising = curr_g > prev_g
        
        if curr_c > 0:  # Surface price is showing Plus (+)
            if g_is_rising:  # Deep cascade sequence is also expanding up -> 10% Real Move
                status_list.append("🟢 CASCADE BULLISH (Strong Trend)")
            else:  # Surface is plus but deep cascade is falling -> 90% CALL TRAP
                status_list.append("⚠️ 90% FAKE UPMOVE (Cascade Alert!)")
        else:  # Surface price is showing Minus (-)
            if not g_is_rising:  # Deep cascade sequence is breaking down -> Real Crash
                status_list.append("🔴 CASCADE BEARISH (Strong Melt)")
            else:  # Surface is minus but deep cascade is absorbing -> 90% PUT TRAP
                status_list.append("⚠️ 90% FAKE DOWNMOVE (Short Covering!)")
                
    df['Signal_Status'] = status_list

    # 🔥 FILTER TRICK: Calculation January se hui, par filter strictly May 1, 2026 se karega
    df_filtered = df[df['Raw_Date'] >= '2026-05-01'].copy()
    
    if not df_filtered.empty:
        # Arrange grid layout with proper sequence of columns
        show_df = df_filtered[['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column F', 'Column G', 'Signal_Status']].copy()
        show_df = show_df.iloc[::-1]  # Latest rows on top
        
        # Grid Theme Color Engine
        def color_trap_grid(val):
            if "🟢" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
            if "🔴" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
            if "⚠️" in val: return 'background-color: #d35400; color: white; font-weight: bold;'
            return ''

        # Dynamic Frame Render with explicit floating formats
        st.dataframe(show_df.style.format({
            'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
            'Column E': '{:.4f}', 'Column F': '{:.4f}', 'Column G': '{:.4f}'
        }).map(color_trap_grid, subset=['Signal_Status']), use_container_width=True)
    else:
        st.warning("May 2026 ka filtered data range khali hai.")
else:
    st.error("Data pipeline load error: Data stream structure couldn't be mounted on Streamlit.")
