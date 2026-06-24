import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SCIENTIFIC UI THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AGCA 2-Year Hourly Freeze", layout="wide")

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
            border: 1px solid #3b82f6;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="decoder-block">
        <h1>🌌 AGCA 2-Year Anchor Engine — Hourly Timeline</h1>
        <p><b>Computational Anchor:</b> 2-Year Yahoo Dataset Freezed | <b>Execution Boundary:</b> Strictly From 01 June 2025<br>
        <b>Asset Matrix:</b> NIFTY 50 Index (^NSEI) | Preserving Only Columns A, B, C, and D Filter Nodes.</p>
    </div>
""", unsafe_allow_html=True)

if 'agca_2y_hourly_db' not in st.session_state:
    st.session_state.agca_2y_hourly_db = pd.DataFrame()

st.sidebar.subheader("🔬 Engine Controls")
run_sync = st.sidebar.button("🔄 Initialize 2025 2-Year Handshake")

# ==============================================================================
# 2. COMPUTATIONAL DATA FREEZE PIPELINE (START: 01 JUNE 2025)
# ==============================================================================
if len(st.session_state.agca_2y_hourly_db) == 0 or run_sync:
    with st.spinner("Pulling 2-Year Dataset and Filtering 1-Hour Candles from 01 June 2025..."):
        try:
            # Step 1: Pull 2 years of raw hourly data to bypass Yahoo's historical limits
            nifty_raw = yf.download(tickers="^NSEI", period="2y", interval="1h", progress=False)
            
            if not nifty_raw.empty:
                if isinstance(nifty_raw.columns, pd.MultiIndex):
                    nifty_raw.columns = [col[0] for col in nifty_raw.columns]
                
                nifty_raw.columns = [str(col).strip().title() for col in nifty_raw.columns]
                nifty_raw.index = pd.to_datetime(nifty_raw.index)
                
                # Step 2: Calculate EMAs on full dataset first to maintain continuous lookback accuracy
                nifty_raw['EMA_20'] = nifty_raw['Close'].ewm(span=20, adjust=False).mean()
                nifty_raw['EMA_50'] = nifty_raw['Close'].ewm(span=50, adjust=False).mean()
                
                # Step 3: CRITICAL TIME FILTER - Slice data strictly from 01 June 2025 onwards
                target_start = pd.to_datetime("2025-06-01").tz_localize(nifty_raw.index.tz)
                filtered_nifty = nifty_raw[nifty_raw.index >= target_start]
                
                if filtered_nifty.empty:
                    st.error("Filtered range returned empty array. Please check date boundaries.")
                else:
                    # Freeze running candle
                    frozen_data = filtered_nifty.iloc[:-1].copy() if len(filtered_nifty) > 1 else filtered_nifty.copy()
                    frozen_data = frozen_data.reset_index()
                    
                    time_key = 'Datetime' if 'Datetime' in frozen_data.columns else frozen_data.columns[0]
                    total_elements = len(frozen_data)
                    
                    col_a = np.array(frozen_data['Close'].values, dtype=float).flatten()
                    ema20_arr = np.array(frozen_data['EMA_20'].values, dtype=float).flatten()
                    ema50_arr = np.array(frozen_data['EMA_50'].values, dtype=float).flatten()
                    
                    col_b = np.zeros(total_elements, dtype=float)
                    col_c = np.zeros(total_elements, dtype=float)
                    col_d_state = []
                    
                    # Step 4: Core Baseline Seeding
                    if total_elements > 0:
                        col_b[0] = float(col_a[0])
                        col_c[0] = 0.0
                        col_d_state.append("Warmup")
                    
                    multiplier = 0.0001
                    
                    # Step 5 to Step 8: Production Math Iteration Loop
                    for t in range(1, total_elements):
                        col_b[t] = col_b[t-1] + (multiplier * (col_a[t] - col_b[t-1]))
                        col_c[t] = col_a[t] - col_b[t]
                        
                        current_sign = np.sign(col_c[t])
                        prev_sign = np.sign(col_c[t-1])
                        
                        current_close = col_a[t]
                        c_ema20 = ema20_arr[t]
                        c_ema50 = ema50_arr[t]
                        
                        # Apply your Exact EMA Confirmation Rules
                        if prev_sign == -1 and current_sign == 1:
                            if current_close > c_ema20 and current_close > c_ema50:
                                col_d_state.append("BULLISH (Above 20/50 EMA)")
                            else:
                                col_d_state.append("No Structure")
                        elif prev_sign == 1 and current_sign == -1:
                            if current_close < c_ema20 and current_close < c_ema50:
                                col_d_state.append("BEARISH (Below 20/50 EMA)")
                            else:
                                col_d_state.append("No Structure")
                        else:
                            col_d_state.append("No Structure")
                    
                    # Step 8: Save to persistent session database
                    st.session_state.agca_2y_hourly_db = pd.DataFrame({
                        'Date_Time': [pd.to_datetime(x).strftime('%d %b %Y %H:%M') for x in frozen_data[time_key].values],
                        'Column A (Nifty Close)': col_a.round(2),
                        'Column B (Anchor)': col_b.round(4),
                        'Column C (Delta Variance)': col_c.round(4),
                        'Column D (EMA Filter State)': col_d_state
                    })
                    st.sidebar.success("2-Year Database Sliced & Frozen!")
            else:
                st.error("Data fetch error from Yahoo Finance.")
        except Exception as ex:
            st.error(f"Core breakdown: {str(ex)}")

# ==============================================================================
# 3. PRESENTATION GRID LAYER (LATEST DATA ON TOP)
# ==============================================================================
output_matrix = st.session_state.agca_2y_hourly_db.copy()

if not output_matrix.empty:
    st.write(f"### 📊 Processed Pricing Stream: **{len(output_matrix)} Hourly Nodes Freezed**")
    inverted_view = output_matrix.iloc[::-1].reset_index(drop=True)
    
    def style_column_d(val):
        if "BULLISH" in str(val):
            return 'background-color: #14532d; color: #a7f3d0; font-weight: bold; border: 1px solid #10b981;'
        elif "BEARISH" in str(val):
            return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold; border: 1px solid #ef4444;'
        return ''

    try:
        styled_df = inverted_view.style.map(style_column_d, subset=['Column D (EMA Filter State)'])
    except AttributeError:
        styled_df = inverted_view.style.applymap(style_column_d, subset=['Column D (EMA Filter State)'])
        
    st.dataframe(styled_df, use_container_width=True)
else:
    st.info("Historical storage empty. Press 'Initialize 2025 2-Year Handshake' to execute.")
