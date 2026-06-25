import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# SYSTEM CALCULATION ENGINE: 8-STEP VERIFICATION MATRIX WITH FIXED START
# ==============================================================================

# STEP 1: SCIENTIFIC UI THEME CONFIGURATION
st.set_page_config(page_title="Nifty Fixed 20-Bar Core", layout="wide")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #020617 !important;
            color: #f1f5f9 !important;
        }
        h1, h3, p, span, label { color: #ffffff !important; }
        .discovery-block {
            background: linear-gradient(135deg, #090d26 0%, #030712 100%);
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #10b981;
            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.2);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="discovery-block">
        <h1>🌌 Nifty 50 Strict Engine: Calculation Starts From 20th Candle</h1>
        <p><b>Strict Rule Matrix:</b> Bars 1-19 Locked Empty ➔ Active Processing & Lookback Injection From Bar 20 Onwards</p>
    </div>
""", unsafe_allow_html=True)

# STEP 2: SIDEBAR CONTROLS REGISTRY
st.sidebar.subheader("🔬 Strict Lookback Parameters")
ticker = st.sidebar.text_input("Target Ticker (Yahoo Finance)", value="^NSEI")
fast_len = st.sidebar.number_input("Fast EMA Length (on C)", min_value=1, value=1)
slow_len = st.sidebar.number_input("Slow EMA Length (on C)", min_value=2, value=5)
signal_len = st.sidebar.number_input("Signal Line Strict Lookback", min_value=2, value=20)

run_sync = st.sidebar.button("🔄 Execute Strict Handshake Loop")

if 'nifty_strict_20_db' not in st.session_state:
    st.session_state.nifty_strict_20_db = pd.DataFrame()

# STEP 3: DATA INGESTION PIPELINE
if len(st.session_state.nifty_strict_20_db) == 0 or run_sync:
    with st.spinner("Processing Strict 20-Bar Lookback Matrix..."):
        try:
            import yfinance as yf
            raw_feed = yf.download(tickers=str(ticker), interval="1h", period="2y", progress=False)
        except ModuleNotFoundError:
            raw_feed = pd.DataFrame()

        # Synthetic Fallback Matrix Core
        if raw_feed.empty:
            date_range = pd.date_range(start="2025-01-01", end="2026-12-31", freq="1h")
            np.random.seed(42)
            simulated_close = 23500 + np.cumsum(np.random.normal(0, 50, len(date_range)))
            raw_feed = pd.DataFrame({'Close': simulated_close}, index=date_range)
            raw_feed.index.name = 'Datetime'

        # STEP 4: VECTOR BASELINE EXTRACTION
        if isinstance(raw_feed.columns, pd.MultiIndex):
            col_a = raw_feed['Close'].values.flatten()
        else:
            col_a = raw_feed.iloc[:, raw_feed.columns.get_loc('Close') if 'Close' in raw_feed.columns else 0].values.flatten()

        raw_dates = raw_feed.index.to_numpy()
        total_elements = len(col_a)

        if total_elements > 0:
            # STEP 5: MATHEMATICAL ARRAY CORE PROCESSING (Strict 1-8 Step Verification Counting)
            col_b = np.zeros(total_elements, dtype=float)         # Kalman Baseline
            col_c = np.zeros(total_elements, dtype=float)         # Column C = A - B
            col_d_macd = np.zeros(total_elements, dtype=float)    # Column D (MACD Line)
            col_e_signal = np.zeros(total_elements, dtype=float)  # Column E (Signal Line)
            col_f_hist = np.zeros(total_elements, dtype=float)    # Column F (Histogram)
            time_list = ["" for _ in range(total_elements)]

            # 1. Compute Kalman Filter directly on Column A
            x_est = col_a[0]
            p_est = 1.0
            q_process = 0.001  
            r_measure = 0.1    

            for t in range(total_elements):
                time_list[t] = pd.to_datetime(raw_dates[t]).strftime('%d %b %Y %H:%M')
                p_prior = p_est + q_process
                k_gain = p_prior / (p_prior + r_measure)
                x_est = x_est + k_gain * (col_a[t] - x_est)
                p_est = (1.0 - k_gain) * p_prior
                col_b[t] = x_est

            # 2. Extract Column C (The core objective vector)
            col_c = col_a - col_b

            # Multiplier configuration
            k_fast = 2.0 / (float(fast_len) + 1.0)
            k_slow = 2.0 / (float(slow_len) + 1.0)
            k_sig = 2.0 / (float(signal_len) + 1.0)

            # 3. STRICT BAR 20 ACTIVATION ENGINE LOOP
            # Pehli 19 candles (index 0 to 18) tak sab kuch flat lock rahega
            fast_ema = 0.0
            slow_ema = 0.0
            
            for t in range(total_elements):
                if t < 19:
                    # Flat zero tracking during warm up state
                    col_d_macd[t] = 0.0
                    col_e_signal[t] = 0.0
                    col_f_hist[t] = 0.0
                elif t == 19:
                    # EXACT 20TH CANDLE (Index 19): Seed value initialization
                    fast_ema = col_c[t]
                    slow_ema = col_c[t]
                    col_d_macd[t] = fast_ema - slow_ema
                    col_e_signal[t] = col_d_macd[t]
                    col_f_hist[t] = 0.0
                else:
                    # Continuous dynamic array calculation from candle 21 onwards
                    fast_ema = (col_c[t] * k_fast) + (fast_ema * (1.0 - k_fast))
                    slow_ema = (col_c[t] * k_slow) + (slow_ema * (1.0 - k_slow))
                    col_d_macd[t] = fast_ema - slow_ema
                    
                    # Signal smoothing on calculated MACD line
                    col_e_signal[t] = (col_d_macd[t] * k_sig) + (col_e_signal[t-1] * (1.0 - k_sig))
                    col_f_hist[t] = col_d_macd[t] - col_e_signal[t]

            # STEP 6: LAPTOP EXCEL ZONE CLASSIFICATION INTEGRATION
            excel_zones = []
            for t in range(total_elements):
                if t < 19:
                    excel_zones.append("🔒 Lookback Squeeze (Calculation Blocked)")
                else:
                    if col_f_hist[t] > 0.0:
                        excel_zones.append("🟢 Active Bullish Grid (SELL PUT)")
                    elif col_f_hist[t] < 0.0:
                        excel_zones.append("🔴 Active Bearish Grid (SELL CALL)")
                    else:
                        excel_zones.append("Balanced Baseline")

            # STEP 7: SECURE TRANSITION PACKING & DATE FREEZE LOCK
            research_df = pd.DataFrame({
                'Date_Time': time_list,
                'Column A (Nifty Close)': [float(x) for x in col_a],
                'Column B (Kalman Filter)': [float(x) for x in col_b],
                'Column C (Pure Delta)': [float(x) for x in col_c],
                'Column D (MACD Line)': [float(x) for x in col_d_macd],
                'Column E (Signal Line)': [float(x) for x in col_e_signal],
                'Column F (Histogram)': [float(x) for x in col_f_hist],
                'Excel Market Zone': excel_zones
            })

            research_df['Datetime_Parsed'] = pd.to_datetime(research_df['Date_Time'], format='%d %b %Y %H:%M')
