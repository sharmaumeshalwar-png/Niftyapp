import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ==============================================================================
# STEP 1: PREMIUM DARK BASE CONFIGURATION (ALL POSSIBLE OUTCOME DATES COMPLIANT)
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
# STEP 2: DATA PIPELINE VIA YFINANCE
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
    # STEP 3: CORE 5-STAGE MATHEMATICAL CASCADE (8-STEP VERIFICATION TRACKING)
    # ==============================================================================
    
    # 3.1 Step 1: Process Baseline Midpoint (Column A)
    df['Column A'] = ((df['High'] + df['Low']) / 2.0).astype(float)
    
    # 3.2 Step 2: Compute Exponential Smooth Cascade 1 (Column B)
    col_b = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_b[0] = float(df['Column A'].values[0])
    for i in range(1, total_rows):
        col_b[i] = col_b[i-1] + (multiplier * (float(df['Column A'].values[i]) - col_b[i-1]))
    df['Column B'] = col_b
    
    # 3.3 Step 3: Compute First Differential Delta Array (Column C)
    df['Column C'] = (df['Column A'] - df['Column B']).astype(float)
    
    # 3.4 Step 4: Compute Exponential Smooth Cascade 2 (Column E)
    col_e = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_e[0] = float(df['Column C'].values[0])
    for i in range(1, total_rows):
        col_e[i] = col_e[i-1] + (multiplier * (float(df['Column C'].values[i]) - col_e[i-1]))
    df['Column E'] = col_e
    
    # 3.5 Step 5: Compute Second Differential Delta Array (Column F)
    df['Column F'] = (df['Column C'] - df['Column E']).astype(float)
    
    # 3.6 Step 6: Compute Exponential Smooth Cascade 3 (Column G)
    col_g = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_g[0] = float(df['Column F'].values[0])
    for i in range(1, total_rows):
        col_g[i] = col_g[i-1] + (multiplier * (float(df['Column F'].values[i]) - col_g[i-1]))
