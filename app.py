import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ==============================================================================
# 1. BASE MATRIX CONFIGURATION (1-HOUR TIME MATRIX)
# ==============================================================================
st.set_page_config(page_title="Nifty Column M Matrix (1-Hour)", layout="wide")

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
            border-left: 5px solid #ec4899;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-block">
        <h1>⚡ Nifty 50 Pure Cascade (1-Hour Candle Matrix Engine)</h1>
        <p><b>Interval:</b> 1 Hour (1H) Candles | <b>Data Depth:</b> Max Allowed Rolling Window<br>
        <b>Column M Matrix Rule:</b> If Column L sign changes, Column M turns <b>Absolute Black</b>. Otherwise, stays <b>Pure White</b>.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA PIPELINE (MAX HOURLY RETRIEVAL SAFEGUARD)
# ==============================================================================
@st.cache_data(ttl=180)
def load_pure_data():
    # '730d' is the absolute maximum allowed range for hourly interval data in Yahoo Finance
    df_raw = yf.download(tickers="^NSEI", period="730d", interval="1h")
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
    # Hour filter formatting including clock time
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    total_rows = len(df)
    mul = 0.0001
    
    # ==============================================================================
    # 3. MATHEMATICAL CASCADE ENGINE (8-STEP VERIFICATION TRACKING)
    # ==============================================================================
    df['Column A'] = ((df['High'] + df['Low']) / 2.0).astype(float)
    
    col_b = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_b[0] = float(df['Column A'].values[0])
    for i in range(1, total_rows):
        col_b[i] = col_b[i-1] + (mul * (float(df['Column A'].values[i]) - col_b[i-1]))
    df['Column B'] = col_b
    
    df['Column C'] = (df['Column A'] - df['Column B']).astype(float)
    
    col_e = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_e[0] = float(df['Column C'].values[0])
    for i in range(1, total_rows):
        col_e[i] = col_e[i-1] + (mul * (float(df['Column C'].values[i]) - col_e[i-1]))
    df['Column E'] = col_e
    
    df['Column F'] = (df['Column C'] - df['Column E']).astype(float)
    
    col_g = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_g[0] = float(df['Column F'].values[0])
    for i in range(1, total_rows):
        col_g[i] = col_g[i-1] + (mul * (float(df['Column F'].values[i]) - col_g[i-1]))
    df['Column G'] = col_g
    
    df['Column H'] = (df['Column F'] - df['Column G']).astype(float)
    
    col_i = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_i[0] = float(df['Column H'].values[0])
    for i in range(1, total_rows):
        col_i[i] = col_i[i-1] + (mul * (float(df['Column H'].values[i]) - col_i[i-1]))
    df['Column I'] = col_i
    
    col_j = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_j[0] = float(col_i[0])
    for i in range(1, total_rows):
        col_j[i] = col_j[i-1] + (mul * (col_i[i] - col_j[i-1]))
    df['Column J'] = col_j
    
    # 4. HOURLY K & L MATRIX CORE
    df['Column K'] = np.sign(df['Column F'].values).astype(float) - np.sign(df['Column H'].values).astype(float)
    
    col_l = np.zeros(total_rows, dtype=float)
    for i in range(1, total_rows):
        col_l[i] = col_i[i] - col_i[i-1]
    df['Column L'] = col_l

    # ==============================================================================
    # 5. COLUMN M HOURLY LOGIC ENGINE
    # ==============================================================================
    m_txt = ["➡️ CONTINUOUS"] * total_rows
    chg_flag = np.zeros(total_rows, dtype=bool)
    last_v = 0
    
    for i in range(1, total_rows):
        v = col_l[i]
        curr_s = 1 if v > 0 else (-1 if v < 0 else 0)
        if curr_s != 0:
            if last_v != 0 and curr_s != last_v:
                chg_flag[i] = True
                m_txt[i] = "🔄 SIGN CHANGED"
            last_v = curr_s
            
    df['Column M'] = m_txt
    df['L_Sign_Change'] = ch
