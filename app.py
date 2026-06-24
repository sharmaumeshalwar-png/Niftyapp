import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import os

warnings.filterwarnings('ignore')

CSV_FILE_PATH = "nifty_2y_hourly_range_theory.csv"

# ==============================================================================
# 1. SCIENTIFIC UI THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AGCA Range Breakout Engine", layout="wide")

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
        <h1>🌌 AGCA Time-Price Range Breakthrough Engine</h1>
        <p><b>Theory Applied:</b> Wyckoff Phase Transition & Range Boundaries | <b>Storage Mode:</b> Local CSV Locked<br>
        Locks High/Low boundaries on Column C sign flips and triggers signals only on absolute physical breakouts.</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.subheader("🔬 Control Panel")
force_download = st.sidebar.button("🔄 Force Download New Data")

# ==============================================================================
# 2. COMPUTATIONAL RANGE BREAKOUT THEORY PIPELINE
# ==============================================================================
data_loaded = False
output_df = pd.DataFrame()

if os.path.exists(CSV_FILE_PATH) and not force_download:
    with st.spinner("Loading Range Theory dataset from Hard Drive..."):
        output_df = pd.read_csv(CSV_FILE_PATH)
        data_loaded = True
else:
    with st.spinner("Compiling Range Transition Matrix..."):
        try:
            nifty_raw = yf.download(tickers="^NSEI", period="2y", interval="1h", progress=False)
            
            if not nifty_raw.empty:
                if isinstance(nifty_raw.columns, pd.MultiIndex):
                    nifty_raw.columns = [col[0] for col in nifty_raw.columns]
                
                nifty_raw.columns = [str(col).strip().title() for col in nifty_raw.columns]
                nifty_raw.index = pd.to_datetime(nifty_raw.index)
                
                nifty_raw['EMA_20'] = nifty_raw['Close'].ewm(span=20, adjust=False).mean()
                nifty_raw['EMA_50'] = nifty_raw['Close'].ewm(span=50, adjust=False).mean()
                
                # Fetch High and Low explicitly to map range boxes
                nifty_raw['Raw_High'] = nifty_raw['High']
                nifty_raw['Raw_Low'] = nifty_raw['Low']
                
                target_start = pd.to_datetime("2025-06-01").tz_localize(nifty_raw.index.tz)
                filtered_nifty = nifty_raw[nifty_raw.index >= target_start]
                
                if not filtered_nifty.empty:
                    frozen_data = filtered_nifty.iloc[:-1].copy() if len(filtered_nifty) > 1 else filtered_nifty.copy()
                    frozen_data = frozen_data.reset_index()
                    
                    time_key = 'Datetime' if 'Datetime' in frozen_data.columns else frozen_data.columns[0]
                    total_elements = len(frozen_data)
                    
                    col_a = np.array(frozen_data['Close'].values, dtype=float).flatten()
                    col_high = np.array(frozen_data['Raw_High'].values, dtype=float).flatten()
                    col_low = np.array(frozen_data['Raw_Low'].values, dtype=float).flatten()
                    
                    ema20_arr = np.array(frozen_data['EMA_20'].values, dtype=float).flatten()
                    ema50_arr = np.array(frozen_data['EMA_50'].values, dtype=float).flatten()
                    
                    col_b = np.zeros(total_elements, dtype=float)
                    col_c = np.zeros(total_elements, dtype=float)
                    
                    if total_elements > 0:
                        col_b[0] = float(col_a[0])
                        col_c[0] = 0.0
                    
                    multiplier = 0.0001
                    
                    # Compute Core Kalan Math
                    for t in range(1, total_elements):
                        col_b[t] = col_b[t-1] + (multiplier * (col_a[t] - col_b[t-1]))
                        col_c[t] = col_a[t] - col_b[t]
                    
                    col_d_state = ["No Structure"] * total_elements
                    
                    # Range Bound Boundary Memory Registers
                    locked_high = 999999.0
                    locked_low = 0.0
                    
                    # Loop execution for Range Breakthrough Strategy
                    for t in range(1, total_elements):
                        current_sign = np.sign(col_c[t])
                        prev_sign = np.sign(col_c[t-1])
                        
                        current_close = col_a[t]
                        c_ema20 = ema20_arr[t]
                        c_ema50 = ema50_arr[t]
                        
                        # Rule 1: Capture range boundary on sign transitions
                        if prev_sign != current_sign:
                            locked_high = float(col_high[t])
                            locked_low = float(col_low[t])
                        
                        # Rule 2: Evaluate Breakout Criteria
                        if current_close > locked_high and current_close > c_ema20 and current_close > c_ema50:
                            col_d_state[t] = "BULLISH BREAKOUT"
                        elif current_close < locked_low and current_close < c_ema20 and current_close < c_ema50:
                            col_d_state[t] = "BEARISH BREAKDOWN"
                        else:
                            col_d_state[t] = "Inside Range Box"
                    
                    output_df = pd.DataFrame({
                        'Date_Time': [pd.to_datetime(x).strftime('%d %b %Y %H:%M') for x in frozen_data[time_key].values],
                        'Column A (Nifty Close)': col_a.round(2),
                        'Column B (Anchor)': col_b.round(4),
                        'Column C (Delta Variance)': col_c.round(4),
                        'Column D (Range Signal State)': col_d_state
                    })
                    
                    output_df.to_csv(CSV_FILE_PATH, index=False)
                    data_loaded = True
        except Exception as ex:
            st.error(f"Theory processing error: {str(ex)}")

# ==============================================================================
# 3. PRESENTATION GRID LAYER
# ==============================================================================
if data_loaded and not output_df.empty:
    st.write(f"### 📊 Wyckoff Transition Board: **{len(output_df)} Intervals Calculated**")
    inverted_view = output_df.iloc[::-1].reset_index(drop=True)
    
    def style_column_d(val):
        if "BULLISH" in str(val):
            return 'background-color: #064e3b; color: #34d399; font-weight: bold;'
        elif "BEARISH" in str(val):
            return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;'
        return 'color: #94a3b8;'

    try:
        styled_df = inverted_view.style.map(style_column_d, subset=['Column D (Range Signal State)'])
    except AttributeError:
        styled_df = inverted_view.style.applymap(style_column_d, subset=['Column D (Range Signal State)'])
        
    st.dataframe(styled_df, use_container_width=True)
else:
    st.info("Local storage raw file offline. Click 'Force Download New Data' to spin up the Range Engine.")
