import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 1. PREMIUM BLACK THEME CONFIGURATION
st.set_page_config(page_title="Nifty Fixed Row Cascade", layout="wide")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0b0f19 !important;
            color: #ffffff !important;
        }
        h1, h3, p, span, label { color: #ffffff !important; }
        .title-block {
            background: linear-gradient(90deg, #111827, #1f2937);
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #ef4444;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-block">
        <h1>🎯 Nifty 50 Pure Cascade (Fixed L = Row 2 - Row 1)</h1>
        <p><b>Column K:</b> (Sign of F) - (Sign of H) | <b>Column L:</b> Strict Row 2 - Row 1 Differential Math<br>
        <b>Rule:</b> Trend Change triggers instantly when price moves opposite to the Column L structural vector.</p>
    </div>
""", unsafe_allow_html=True)

# 2. DATA PIPELINE WITH TRAILING BUFFER
@st.cache_data(ttl=300)
def load_pure_data():
    df_raw = yf.download(tickers="^NSEI", period="2y", interval="1h")
    if df_raw.empty:
        return pd.DataFrame()
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    return df_raw

df = load_pure_data()

if not df.empty:
    df = df.reset_index()
    
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    total_rows = len(df)
    multiplier = 0.0001
    
    # 3. CORE 5-STAGE MATH CASCADE
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
    
    # 4. COLUMN K CALCULATOR
    sign_f = np.sign(df['Column F'].values).astype(float)
    sign_h = np.sign(df['Column H'].values).astype(float)
    df['Column K'] = sign_f - sign_h
    
    # 🛠️ FIXED COLUMN L: Strict Row Forward Calculation (Row 2 - Row 1)
    col_l = np.zeros(total_rows, dtype=float)
    for i in range(1, total_rows):
        col_l[i] = col_i[i] - col_i[i-1]  # Nayi row values minus purani row values
    df['Column L'] = col_l
    
    # 5. STRICT OPPOSITE TREND ENGINE (8-Step Verified Matrix)
    status_list = ["System Booting"]
    for i in range(1, total_rows):
        curr_l = float(df['Column L'].values[i])
        price_move = float(df['Column A'].values[i]) - float(df['Column A'].values[i-1])
        curr_k = float(df['Column K'].values[i])
        
        # Core Trap Condition: Price action breaks completely opposite to Row Vector L
        price_opposes_l = (curr_l > 0.0 and price_move < 0.0) or (curr_l < 0.0 and price_move > 0.0)
        
        if price_opposes_l:
            status_list.append("⚡ TREND CHANGE ALERT")
        else:
            if curr_k == 2.0 or curr_k == 0.0:
                status_list.append("🟢 SIGN BULLISH")
            elif curr_k == -2.0 or curr_k == 0.0:
                status_list.append("🔴 SIGN BEARISH")
            else:
                status_list.append("⚠️ INVERSION ALERT")
                
    df['Signal_Status'] = status_list

    # 6. FILTER AND INVERT VIEW (Jan 1, 2025 Layout Alignment)
    df_filtered = df[df['Raw_Date'] >= '2025-01-01'].copy()
    
    if not df_filtered.empty:
        show_df = df_filtered[['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column F', 'Column G', 'Column H', 'Column I', 'Column J', 'Column K', 'Column L', 'Signal_Status']].copy()
        show_df = show_df.iloc[::-1]  # Latest candle on top (Screenshot style matching)
        
        # Dark Grid CSS Formatter Engine
        def color_trap_grid(val):
            if "⚡ TREND" in val: return 'background-color: #7b1113; color: #ffffff; font-weight: bold; border: 1.5px solid yellow;'
            if "🟢" in val: return 'background-color: #064e3b; color: #34d399; font-weight: bold;'
            if "🔴" in val: return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;'
            return 'background-color: #1f2937; color: #d1d5db;'

        # Clean Table Output Render
        st.dataframe(show_df.style.format({
            'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
            'Column E': '{:.4f}', 'Column F': '{:.4f}', 'Column G': '{:.4f}',
            'Column H': '{:.4f}', 'Column I': '{:.4f}', 'Column J': '{:.4f}', 
            'Column K': '{:.0f}', 'Column L': '{:.6f}'
        }).map(color_trap_grid, subset=['Signal_Status']), use_container_width=True)
    else:
        st.warning("January 1, 2025 filtered range empty.")
else:
    st.error("Data pipeline load error.")
