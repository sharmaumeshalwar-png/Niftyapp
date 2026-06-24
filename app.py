import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import os

warnings.filterwarnings('ignore')

CSV_FILE_PATH = "nifty_2y_hourly_difference_core.csv"

# ==============================================================================
# 1. SCIENTIFIC UI THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AGCA Difference Engine", layout="wide")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #020617 !important;
            color: #f1f5f9 !important;
        }
        h1, h3, p, span, label { color: #f8fafc !important; }
        .decoder-block {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #f43f5e;
            box-shadow: 0 4px 20px rgba(244, 63, 94, 0.15);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="decoder-block">
        <h1>🌌 AGCA Gaussian-Kalman Variance Difference Matrix</h1>
        <p><b>Column C:</b> Kalan Delta | <b>Column D:</b> 2-Pole Gaussian | <b>Column E:</b> Kalman Filter Line<br>
        <b>Column F:</b> Spread Node (Column D - Column E) | <b>Column G:</b> Core Signal Block</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.subheader("🔬 Control Room")
force_download = st.sidebar.button("🔄 Force Sync Difference Matrix")

# ==============================================================================
# 2. SEAMLESS DUAL FILTER SPREAD PIPELINE
# ==============================================================================
data_loaded = False
output_df = pd.DataFrame()

if os.path.exists(CSV_FILE_PATH) and not force_download:
    with st.spinner("Reading stable difference matrix from Hard Drive..."):
        try:
            output_df = pd.read_csv(CSV_FILE_PATH)
            data_loaded = True
        except:
            data_loaded = False

if not data_loaded:
    with st.spinner("Compiling multi-layer vector arrays..."):
        try:
            nifty_raw = yf.download(tickers="^NSEI", period="2y", interval="1h", progress=False)
            
            if not nifty_raw.empty:
                if isinstance(nifty_raw.columns, pd.MultiIndex):
                    nifty_raw.columns = [col[0] for col in nifty_raw.columns]
                
                nifty_raw.columns = [str(col).strip().title() for col in nifty_raw.columns]
                nifty_raw.index = pd.to_datetime(nifty_raw.index)
                
                # EMAs Base Injection
                nifty_raw['EMA_20'] = nifty_raw['Close'].ewm(span=20, adjust=False).mean()
                nifty_raw['EMA_50'] = nifty_raw['Close'].ewm(span=50, adjust=False).mean()
                
                # Slicing Strictly from 01 June 2025
                target_start = pd.to_datetime("2025-06-01").tz_localize(nifty_raw.index.tz)
                filtered_nifty = nifty_raw[nifty_raw.index >= target_start]
                
                if not filtered_nifty.empty:
                    frozen_data = filtered_nifty.iloc[:-1].copy() if len(filtered_nifty) > 1 else filtered_nifty.copy()
                    frozen_data = frozen_data.reset_index()
                    
                    time_key = 'Datetime' if 'Datetime' in frozen_data.columns else frozen_data.columns[0]
                    total_elements = len(frozen_data)
                    
                    col_a = np.array(frozen_data['Close'].values, dtype=float).flatten()
                    ema20_arr = np.array(frozen_data['EMA_20'].values, dtype=float).flatten()
                    ema50_arr = np.array(frozen_data['EMA_50'].values, dtype=float).flatten()
                    
                    col_b = np.zeros(total_elements, dtype=float)
                    col_c = np.zeros(total_elements, dtype=float)
                    
                    if total_elements > 0:
                        col_b[0] = float(col_a[0])
                        col_c[0] = 0.0
                    
                    multiplier = 0.0001
                    
                    # 1. Kalan Formula Array Iteration
                    for t in range(1, total_elements):
                        col_b[t] = col_b[t-1] + (multiplier * (col_a[t] - col_b[t-1]))
                        col_c[t] = col_a[t] - col_b[t]
                    
                    # 2. 2-Pole Gaussian Calculation
                    col_d = np.zeros(total_elements, dtype=float)
                    length = 14
                    beta = (1 - np.cos(2 * np.pi / length)) / (np.sqrt(2) - 1)
                    alpha = -beta + np.sqrt(beta**2 + 2*beta)
                    
                    c1 = alpha ** 2
                    c2 = 2 * (1 - alpha)
                    c3 = - (1 - alpha) ** 2
                    
                    if total_elements > 1:
                        col_d[0] = col_c[0]
                        col_d[1] = col_c[1]
                    
                    for t in range(2, total_elements):
                        col_d[t] = (c1 * col_c[t]) + (c2 * col_d[t-1]) + (c3 * col_d[t-2])
                    
                    # 3. Kalman Filter on Column D
                    col_e = np.zeros(total_elements, dtype=float)
                    post_error_var = 1.0
                    process_var = 0.01
                    measurement_var = 0.1
                    
                    if total_elements > 0:
                        col_e[0] = col_d[0]
                    
                    for t in range(1, total_elements):
                        prior_estimate = col_e[t-1]
                        prior_error_var = post_error_var + process_var
                        kalman_gain = prior_error_var / (prior_error_var + measurement_var)
                        col_e[t] = prior_estimate + kalman_gain * (col_d[t] - prior_estimate)
                        post_error_var = (1 - kalman_gain) * prior_error_var
                    
                    # 4. NEW LOGIC: Column F = Column D - Column E
                    col_f = col_d - col_e
                    
                    # 5. Signal Execution State (Column G)
                    col_g_state = []
                    for t in range(total_elements):
                        c_close = col_a[t]
                        c_kalan = col_c[t]
                        spread_val = col_f[t]
                        c_ema20 = ema20_arr[t]
                        c_ema50 = ema50_arr[t]
                        
                        if c_kalan > 0 and spread_val > 0 and c_close > c_ema20 and c_close > c_ema50:
                            col_g_state.append("BULLISH WAVE")
                        elif c_kalan < 0 and spread_val < 0 and c_close < c_ema20 and c_close < c_ema50:
                            col_g_state.append("BEARISH WAVE")
                        else:
                            col_g_state.append("No Structure")
                    
                    # Dataframe Compilation
                    output_df = pd.DataFrame({
                        'Date_Time': [pd.to_datetime(x).strftime('%d %b %Y %H:%M') for x in frozen_data[time_key].values],
                        'Column A (Close)': col_a.round(2),
                        'Column B (Anchor)': col_b.round(4),
                        'Column C (Kalan Delta)': col_c.round(4),
                        'Column D (Gaussian)': col_d.round(4),
                        'Column E (Kalman)': col_e.round(4),
                        'Column F (D - E Spread)': col_f.round(4),
                        'Column G (Signal State)': col_g_state
                    })
                    
                    output_df.to_csv(CSV_FILE_PATH, index=False)
                    data_loaded = True
        except Exception as ex:
            st.error(f"Core execution error: {str(ex)}")

# ==============================================================================
# 3. PRESENTATION GRID LAYER
# ==============================================================================
if data_loaded and not output_df.empty:
    st.write(f"### 📊 Processed Spread Board: **{len(output_df)} Timestamps Freezed**")
    inverted_view = output_df.iloc[::-1].reset_index(drop=True)
    
    def style_column_g(val):
        if "BULLISH" in str(val):
            return 'background-color: #064e3b; color: #34d399; font-weight: bold;'
        elif "BEARISH" in str(val):
            return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;'
        return 'color: #475569;'

    try:
        styled_df = inverted_view.style.map(style_column_g, subset=['Column G (Signal State)'])
    except AttributeError:
        styled_df = inverted_view.style.applymap(style_column_g, subset=['Column G (Signal State)'])
        
    st.dataframe(styled_df, use_container_width=True)
else:
    st.info("Local storage empty. Click 'Force Sync Difference Matrix' to initialize calculation frames.")
