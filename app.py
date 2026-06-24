import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import os

warnings.filterwarnings('ignore')

# HARD DRIVE FILE PATH
CSV_FILE_PATH = "nifty_2y_hourly_data.csv"

# ==============================================================================
# 1. SCIENTIFIC UI THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AGCA Hard-Drive Locked Engine", layout="wide")

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
            border: 1px solid #e11d48;
            box-shadow: 0 4px 20px rgba(225, 29, 72, 0.15);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="decoder-block">
        <h1>🌌 AGCA Hard-Drive Locked Engine — 1-Hour Resolution</h1>
        <p><b>Storage Mode:</b> Local CSV Database Lock (0% Data Loss On Code Changes)<br>
        <b>Timeline Target:</b> From 01 June 2025 to 2026 | Preserving Column A, B, C, and D Filter Setup.</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.subheader("🔬 Data Controls")
force_download = st.sidebar.button("🔄 Force Download New Data")

# ==============================================================================
# 2. SEAMLESS HARD-DRIVE DATA INTERACTION PIPELINE
# ==============================================================================
data_loaded = False
output_df = pd.DataFrame()

# Check if the file already exists in your computer's hard drive
if os.path.exists(CSV_FILE_PATH) and not force_download:
    with st.spinner("Loading frozen 2-Year dataset directly from Hard Drive..."):
        output_df = pd.read_csv(CSV_FILE_PATH)
        data_loaded = True
else:
    with st.spinner("Downloading 2-Year Dataset from Yahoo and Freezing to Hard Drive..."):
        try:
            nifty_raw = yf.download(tickers="^NSEI", period="2y", interval="1h", progress=False)
            
            if not nifty_raw.empty:
                if isinstance(nifty_raw.columns, pd.MultiIndex):
                    nifty_raw.columns = [col[0] for col in nifty_raw.columns]
                
                nifty_raw.columns = [str(col).strip().title() for col in nifty_raw.columns]
                nifty_raw.index = pd.to_datetime(nifty_raw.index)
                
                # Calculating Technical EMAs
                nifty_raw['EMA_20'] = nifty_raw['Close'].ewm(span=20, adjust=False).mean()
                nifty_raw['EMA_50'] = nifty_raw['Close'].ewm(span=50, adjust=False).mean()
                
                # Slicing from 01 June 2025 strictly
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
                    col_d_state = []
                    
                    if total_elements > 0:
                        col_b[0] = float(col_a[0])
                        col_c[0] = 0.0
                        col_d_state.append("Warmup")
                    
                    multiplier = 0.0001
                    
                    for t in range(1, total_elements):
                        col_b[t] = col_b[t-1] + (multiplier * (col_a[t] - col_b[t-1]))
                        col_c[t] = col_a[t] - col_b[t]
                        
                        current_sign = np.sign(col_c[t])
                        prev_sign = np.sign(col_c[t-1])
                        
                        current_close = col_a[t]
                        c_ema20 = ema20_arr[t]
                        c_ema50 = ema50_arr[t]
                        
                        # Strict EMA Confirmation Filter Rules
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
                    
                    # Create Dataframe
                    output_df = pd.DataFrame({
                        'Date_Time': [pd.to_datetime(x).strftime('%d %b %Y %H:%M') for x in frozen_data[time_key].values],
                        'Column A (Nifty Close)': col_a.round(2),
                        'Column B (Anchor)': col_b.round(4),
                        'Column C (Delta Variance)': col_c.round(4),
                        'Column D (EMA Filter State)': col_d_state
                    })
                    
                    # HARD DRIVE SAVE: Write data to local machine
                    output_df.to_csv(CSV_FILE_PATH, index=False)
                    st.sidebar.success("Data successfully locked inside Hard Drive!")
                    data_loaded = True
        except Exception as ex:
            st.error(f"Download Breakdown: {str(ex)}")

# ==============================================================================
# 3. PRESENTATION GRID LAYER (LATEST DATA ON TOP)
# ==============================================================================
if data_loaded and not output_df.empty:
    st.write(f"### 📊 Locked Hard-Drive Node: **{len(output_df)} Hourly Intervals Processed**")
    inverted_view = output_df.iloc[::-1].reset_index(drop=True)
    
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
    st.info("No local database file detected. Please click 'Force Download New Data' to initialize the hardware lock.")
