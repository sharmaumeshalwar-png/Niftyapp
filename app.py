import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ==============================================================================
# 1. PREMIUM DARK BASE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Nifty Column M Matrix", layout="wide")

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
        <h1>🎯 Nifty 50 Pure Cascade (Column M Theme Controller)</h1>
        <p><b>Column K:</b> (Sign of F) - (Sign of H) | <b>Column L:</b> (I row 2) - (I row 1) Value Only<br>
        <b>Column M Matrix Rule:</b> If Column L sign changes, Column M turns <b>Absolute Black</b>. Otherwise, stays <b>Pure White</b>.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA PIPELINE VIA YFINANCE
# ==============================================================================
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
    
    # ==============================================================================
    # 3. CORE 5-STAGE MATHEMATICAL CASCADE
    # ==============================================================================
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
    
    # 4. COLUMN K MATRIX MATHEMATICS
    sign_f = np.sign(df['Column F'].values).astype(float)
    sign_h = np.sign(df['Column H'].values).astype(float)
    df['Column K'] = sign_f - sign_h
    
    # 5. COLUMN L MATHEMATICS
    col_l = np.zeros(total_rows, dtype=float)
    for i in range(1, total_rows):
        col_l[i] = col_i[i] - col_i[i-1]
    df['Column L'] = col_l

    # 6. COLUMN M ENGINE & TRACKER MATRIX RULE
    col_m_text = ["➡️ CONTINUOUS"] * total_rows
    l_change_flag = np.zeros(total_rows, dtype=bool)
    
    last_valid_sign = 0
    for i in range(1, total_rows):
        current_val = col_l[i]
        
        if current_val > 0:
            current_sign = 1
        elif current_val < 0:
            current_sign = -1
        else:
            current_sign = 0
            
        if current_sign != 0:
            if last_valid_sign != 0 and current_sign != last_valid_sign:
                l_change_flag[i] = True
                col_m_text[i] = "🔄 SIGN CHANGED"
            last_valid_sign = current_sign
            
    df['Column M'] = col_m_text
    df['L_Sign_Change'] = l_change_flag

    # 7. SIGNAL TRAP AND TREND MATRIX GENERATOR
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

    # ==============================================================================
    # 8. SAFE STYLING & RENDERING PIPELINE (JAN 2026 FILTER)
    # ==============================================================================
    df_filtered = df[df['Raw_Date'] >= '2026-01-01'].copy()
    
    if not df_filtered.empty:
        # Columns select karein jo final UI par show karni hain
        display_columns = ['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column F', 'Column G', 'Column H', 'Column I', 'Column J', 'Column K', 'Column L', 'Column M', 'Signal_Status']
        
        # Chronological inversion (latest data on top)
        show_df = df_filtered[display_columns].copy()
        show_df = show_df.iloc[::-1].reset_index(drop=True)
        
        # Alignment flags maintain karne ke liye filter se match karta hua array nikalna
        change_flags_inverted = df_filtered['L_Sign_Change'].iloc[::-1].reset_index(drop=True)

        # Fail-safe Cell Mapping (Bina axis=1 crash ke chalne wala design)
        def apply_row_styles(row):
            idx = row.name
            styles = [''] * len(row)
            
            # Column M Matrix Styles apply karein
            m_idx = row.index.get_loc('Column M')
            if change_flags_inverted.iloc[idx] == True:
                styles[m_idx] = 'background-color: #000000 !important; color: #ffffff !important; font-weight: bold; border: 1.5px solid #3b82f6;'
            else:
                styles[m_idx] = 'background-color: #ffffff !important; color: #000000 !important; font-weight: bold;'
                
            # Signal Status Styles apply karein
            sig_idx = row.index.get_loc('Signal_Status')
            val = str(row['Signal_Status'])
            if "🟢" in val: 
                styles[sig_idx] = 'background-color: #064e3b; color: #34d399; font-weight: bold;'
            elif "🔴" in val: 
                styles[sig_idx] = 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;'
            elif "⚠️" in val: 
                styles[sig_idx] = 'background-color: #b45309; color: #fef08a; font-weight: bold;'
            else:
                styles[sig_idx] = 'background-color: #1f2937; color: #d1d5db;'
                
            return styles

        # Final Grid Render Engine
        st.dataframe(
            show_df.style.format({
                'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
                'Column E': '{:.4f}', 'Column F': '{:.4f}', 'Column G': '{:.4f}',
                'Column H': '{:.4f}', 'Column I': '{:.4f}', 'Column J': '{:.4f}', 
                'Column K': '{:.0f}', 'Column L': '{:.6f}'
            }).apply(apply_row_styles, axis=1),
            use_container_width=True
        )
    else:
        st.warning("January 1, 202
