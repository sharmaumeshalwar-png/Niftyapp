import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty Pure F-H Sign Matrix", layout="wide")
st.title("🎯 Nifty 50 5-Stage Pure Sign Cascade System (F - H)")
st.write("Column K is strictly calculated as: (Sign of F) - (Sign of H). Running continuously from January 1, 2025.")

# Robust Data Fetcher to ensure continuous history without blank screens
@st.cache_data(ttl=300)
def load_pure_data():
    # Fetching trailing 2 years to bypass any API freeze and get heavy historical padding
    df_raw = yf.download(tickers="^NSEI", period="2y", interval="1h")
    
    if df_raw.empty:
        return pd.DataFrame()
        
    # Flatten MultiIndex columns if present
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
    
    # 3. Column C: Pure Deviation -> A - B (FIXED TYPO HERE)
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
    
    # 🛠️ 10. COLUMN K: STRICTLY SIGN SUBTRACTION -> (Sign of F) - (Sign of H)
    sign_f = np.sign(df['Column F'].values).astype(float)
    sign_h = np.sign(df['Column H'].values).astype(float)
    df['Column K'] = sign_f - sign_h
    
    # 🌟 SIGN-DIFFERENCE LOOP VERIFICATION ENGINE (F - H Match)
    status_list = ["System Booting"]
    for i in range(1, total_rows):
        curr_k = float(df['Column K'].values[i])
        curr_c = float(df['Column C'].values[i])
        
        if curr_c > 0:  # Surface price is showing Plus (+)
            if curr_k == 2.0 or curr_k == 0.0:  
                status_list.append("🟢 SIGN BULLISH (F-H Confirmed)")
            else:  
                status_list.append("⚠️ 90% CALL TRAP (F-H Inversion Alert!)")
        else:  # Surface price is showing Minus (-)
            if curr_k == -2.0 or curr_k == 0.0:  
                status_list.append("🔴 SIGN BEARISH (F-H Confirmed)")
            else:  
                status_list.append("⚠️ 90% PUT TRAP (F-H Inversion Alert!)")
                
    df['Signal_Status'] = status_list

    # 🔥 PRESENTATION FILTER: UI displays strictly from January 1, 2025
    df_filtered = df[df['Raw_Date'] >= '2025-01-01'].copy()
    
    if not df_filtered.empty:
        show_df = df_filtered[['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column F', 'Column G', 'Column H', 'Column I', 'Column J', 'Column K', 'Signal_Status']].copy()
        show_df = show_df.iloc[::-1]  # Latest candle on top
        
        # Grid Theme Color Engine
        def color_trap_grid(val):
            if "🟢" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
            if "🔴" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
            if "⚠️" in val: return 'background-color: #d35400; color: white; font-weight: bold;'
            return ''

        # Dynamic Frame Render
        st.dataframe(show_df.style.format({
            'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
            'Column E': '{:.4f}', 'Column F': '{:.4f}', 'Column G': '{:.4f}',
            'Column H': '{:.4f}', 'Column I': '{:.4f}', 'Column J': '{:.4f}', 
            'Column K': '{:.0f}'  
        }).map(color_trap_grid, subset=['Signal_Status']), use_container_width=True)
    else:
        st.warning("January 1, 2025 filtered range empty.")
else:
    st.error("Data pipeline load error: Data stream structure couldn't be mounted on Streamlit.")
