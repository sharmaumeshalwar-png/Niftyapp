import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 1. PREMIUM DARK BASE CONFIGURATION
st.set_page_config(page_title="Nifty 2026 Stable Matrix", layout="wide")

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
            border-left: 5px solid #3b82f6;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-block">
        <h1>🎯 Nifty 50 Pure Cascade (2026 Fixed Core)</h1>
        <p><b>Data Target:</b> Strictly from 1 Jan 2026 onwards | <b>Column L:</b> (I row 2) - (I row 1) Pure Value<br>
        <b>Column M Matrix Rule:</b> Turns <b>Absolute Black</b> with White Text on strict sign change. Otherwise, stays <b>Pure White</b>.</p>
    </div>
""", unsafe_allow_html=True)

# 2. DATA PIPELINE WITH SAFE HISTORICAL BUFFER
@st.cache_data(ttl=300)
def load_pure_data():
    # Fetching 2 years data to ensure EMA cascade starts with accurate initialization
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
    
    # 3. CORE 5-STAGE MATHEMATICAL CASCADE (Step 1 to 8 Counting)
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
    
    # 🛠️ FIXED CORE MATH FOR COLUMN K (FIXED CRASH)
    sign_f = np.sign(df['Column F'].values).astype(float)
    sign_h = np.sign(df['Column H'].values).astype(float)
    df['Column K'] = sign_f - sign_h
    
    # 4. COLUMN L MATH: Strict Chronological Delta
    col_l = np.zeros(total_rows, dtype=float)
    for i in range(1, total_rows):
        col_l[i] = col_i[i] - col_i[i-1]
    df['Column L'] = col_l

    # 5. COLUMN M STATE ENGINE & TRUE SIGN DETECTION
    col_m_text = ["➡️ CONTINUOUS"] * total_rows
    l_change_flag = np.zeros(total_rows, dtype=bool)
    
    last_known_sign = 0
    for i in range(1, total_rows):
        current_val = col_l[i]
        
        if current_val > 0:
            current_sign = 1
        elif current_val < 0:
            current_sign = -1
        else:
            current_sign = 0
            
        if current_sign != 0:
            if last_known_sign != 0 and current_sign != last_known_sign:
                l_change_flag[i] = True
                col_m_text[i] = "🔄 SIGN CHANGED"
            last_known_sign = current_sign
            
    df['Column M'] = col_m_text
    df['L_Sign_Change'] = l_change_flag

    # SIGNAL STATUS GENERATOR
    status_list = ["System Booting"]
    for i in range(1, total_rows):
        curr_k = float(df['Column K'].values[i])
        curr_c = float(df['Column C'].values[i])
        if curr_c > 0:
            if curr_k == 2.0 or curr_k == 0.0:  status_list.append("🟢 SIGN BULLISH")
            else:  status_list.append("⚠️ 90% CALL TRAP")
        else:
            if curr_k == -2.0 or curr_k == 0.0:  status_list.append("🔴 SIGN BEARISH")
            else:  status_list.append("⚠️ 90% PUT TRAP")
    df['Signal_Status'] = status_list

    # 🌟 6. STRICT FILTER FROM 1 JAN 2026 & VISUAL FLIP (Latest on top)
    df_filtered = df[df['Raw_Date'] >= '2026-01-01'].copy()
    
    if not df_filtered.empty:
        show_df = df_filtered[['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column F', 'Column G', 'Column H', 'Column I', 'Column J', 'Column K', 'Column L', 'Column M', 'L_Sign_Change', 'Signal_Status']].copy()
        show_df = show_df.iloc[::-1].reset_index(drop=True)  # Secure layout inversion
        
        # 7. THEME RENDERING LAYERS
        def style_signal_cells(val):
            if "🟢" in val: return 'background-color: #064e3b; color: #34d399; font-weight: bold;'
            if "🔴" in val: return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;'
            if "⚠️" in val: return 'background-color: #b45309; color: #fef08a; font-weight: bold;'
            return 'background-color: #1f2937; color: #d1d5db;'

        def apply_column_m_theme(row):
            styles = [''] * len(row)
            m_index = row.index.get_loc('Column M')
            
            # If backend flagged true sign change -> Make Column M Absolute Black
            if row['L_Sign_Change'] == True:
                styles[m_index] = 'background-color: #000000 !important; color: #ffffff !important; font-weight: bold; border: 1.5px solid #3b82f6;'
            else:
                # No change -> Pure White with Black Text
                styles[m_index] = 'background-color: #ffffff !important; color: #000000 !important; font-weight: bold;'
            return styles

        # 8. RENDER IMMUTABLE GRID FRAME
        st.dataframe(
            show_df.style.format({
                'Column A': '{:.2f}', 'Column B': '{:.
