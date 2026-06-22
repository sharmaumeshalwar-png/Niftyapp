import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

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
        <h1>⚡ Nifty 50 Pure Cascade (1-Hour Full-Proof Matrix)</h1>
        <p><b>Interval:</b> 1 Hour (1H) Candles | <b>St_List Error Fixed</b><br>
        <b>Column M Matrix Rule:</b> If Column L sign changes, Column M turns <b>Absolute Black</b>. Otherwise, stays <b>Pure White</b>.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA PIPELINE
# ==============================================================================
@st.cache_data(ttl=120, show_spinner=False)
def load_pure_data():
    try:
        df_raw = yf.download(tickers="^NSEI", period="730d", interval="1h", progress=False)
        if df_raw.empty:
            return pd.DataFrame()
        
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        else:
            df_raw.columns = [str(col).strip() for col in df_raw.columns]
            
        df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
        return df_raw
    except Exception as e:
        return pd.DataFrame()

df = load_pure_data()

if not df.empty:
    df = df.reset_index()
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    total_rows = len(df)
    mul = 0.0001
    
    # ==============================================================================
    # 3. MATHEMATICAL CASCADE ENGINE
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
    
    df['Column K'] = np.sign(df['Column F'].values).astype(float) - np.sign(df['Column H'].values).astype(float)
    
    col_l = np.zeros(total_rows, dtype=float)
    for i in range(1, total_rows):
        col_l[i] = col_i[i] - col_i[i-1]
    df['Column L'] = col_l

    # 4. COLUMN M MATRIX
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
    df['L_Sign_Change'] = chg_flag

    # 5. SIGNAL ALIGNED MATRIX
    sig = ["System Booting"]
    for i in range(1, total_rows):
        k = float(df['Column K'].values[i])
        c = float(df['Column C'].values[i])
        l_curr = float(col_l[i])
        l_prev = float(col_l[i-1]) if i > 0 else 0.0
        
        if c > 0:
            if k in [2.0, 0.0]:
                sig.append("🟢 SIGN BULLISH")
            else:
                if l_curr > 0 and l_curr <= l_prev:
                    sig.append("⚠️ CONFIRMED CALL TRAP")
                else:
                    sig.append("🚀 GENUINE UP-BREAKOUT (10%)")
        else:
            if k in [-2.0, 0.0]:
                sig.append("🔴 SIGN BEARISH")
            else:
                if l_curr < 0 and l_curr >= l_prev:
                    sig.append("⚠️ CONFIRMED PUT TRAP")
                else:
                    sig.append("💥 GENUINE DOWN-BREAKOUT (10%)")
                    
    df['Signal_Status'] = sig

    # ==============================================================================
    # 6. FIXED STYLING PIPELINE (DYNAMIC LENGTH BOUNDS)
    # ==============================================================================
    df_f = df.copy()
    cols = ['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column F', 'Column G', 'Column H', 'Column I', 'Column J', 'Column K', 'Column L', 'Column M', 'Signal_Status']
    
    show_df = df_f[cols].copy().iloc[::-1].reset_index(drop=True)
    flags = df_f['L_Sign_Change'].iloc[::-1].reset_index(drop=True)

    def grid_style(row):
        # dynamic array logic completely fixes 'invalid st_list' error
        st_list = [''] * len(row)
        idx = row.name
        
        # Safe boundary target allocation
        m_pos = row.index.get_loc('Column M') if 'Column M' in row.index else -1
        s_pos = row.index.get_loc('Signal_Status') if 'Signal_Status' in row.index else -1
        
        if m_pos != -1:
            if idx < len(flags) and flags.iloc[idx]:
                st_list[m_pos] = 'background-color: #000000 !important; color: #ffffff !important; font-weight: bold; border: 1.5px solid #3b82f6;'
            else:
                st_list[m_pos] = 'background-color: #ffffff !important; color: #000000 !important; font-weight: bold;'
                
        if s_pos != -1:
            val = str(row['Signal_Status'])
            if "🟢" in val: st_list[s_pos] = 'background-color: #064e3b; color: #34d399; font-weight: bold;'
            elif "🔴" in val: st_list[s_pos] = 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;'
            elif "⚠️" in val: st_list[s_pos] = 'background-color: #b45309; color: #fef08a; font-weight: bold;'
            elif "🚀" in val or "💥" in val: st_list[s_pos] = 'background-color: #1e3a8a; color: #93c5fd; font-weight: bold;'
            else: st_list[s_pos] = 'background-color: #1f2937; color: #d1d5db;'
            
        return st_list

    st.dataframe(
        show_df.style.format({
            'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
            'Column E': '{:.4f}', 'Column F': '{:.4f}', 'Column G': '{:.4f}',
            'Column H': '{:.4f}', 'Column I': '{:.4f}', 'Column J': '{:.4f}', 
            'Column K': '{:.0f}', 'Column L': '{:.6f}'
        }).apply(grid_style, axis=1),
        use_container_width=True
    )
else:
    st.error("Data pipeline load error.")
