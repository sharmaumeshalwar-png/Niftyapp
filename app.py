import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# SYSTEM CALCULATION ENGINE: 8-STEP VERIFICATION MATRIX (1 JAN 2026 TIMELINE LOCK)
# ==============================================================================

# STEP 1: SCIENTIFIC UI THEME CONFIGURATION
st.set_page_config(page_title="Nifty Fixed 20-Bar Core 2026", layout="wide")

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
        <h1>🌌 Nifty 50 Strict Engine: Calculation Starts From 20th Candle (2026 Live Timeline)</h1>
        <p><b>System Core:</b> Data Stream Strictly Displayed From 1 Jan 2026 Onwards | Full Array Warm-Up Enabled</p>
    </div>
""", unsafe_allow_html=True)

# STEP 2: SIDEBAR CONTROLS REGISTRY
st.sidebar.subheader("🔬 Strict Lookback Parameters")
ticker = st.sidebar.text_input("Target Ticker (Yahoo Finance)", value="^NSEI")
fast_len = st.sidebar.number_input("Fast EMA Length (on C)", min_value=1, value=1)
slow_len = st.sidebar.number_input("Slow EMA Length (on C)", min_value=2, value=5)
signal_len = st.sidebar.number_input("Signal Line Strict Lookback", min_value=2, value=20)

run_sync = st.sidebar.button("🔄 Execute 2026 Strict Loop")

if 'nifty_strict_2026_db' not in st.session_state:
    st.session_state.nifty_strict_2026_db = pd.DataFrame()

# STEP 3: DATA INGESTION PIPELINE (Downloads back-data for safe math initialization)
if len(st.session_state.nifty_strict_2026_db) == 0 or run_sync:
    with st.spinner("Processing Strict 2026 Lookback Matrix..."):
        try:
            import yfinance as yf
            # Ingesting from late 2025 so that Jan 1, 2026 rows have solid warm-up lines already generated
            raw_feed = yf.download(tickers=str(ticker), interval="1h", start="2025-11-01", progress=False)
        except Exception:
            raw_feed = pd.DataFrame()

        # Synthetic Fallback Matrix Core if API times out
        if raw_feed.empty:
            date_range = pd.date_range(start="2025-11-01", end="2026-12-31", freq="1h")
            np.random.seed(42)
            simulated_close = 23500 + np.cumsum(np.random.normal(0, 45, len(date_range)))
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
            # STEP 5: MATHEMATICAL ARRAY CORE PROCESSING
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
                time_list[t] = pd.to_datetime(raw_dates[t]).strftime('%Y-%m-%d %H:%M')
                p_prior = p_est + q_process
                k_gain = p_prior / (p_prior + r_measure)
                x_est = x_est + k_gain * (col_a[t] - x_est)
                p_est = (1.0 - k_gain) * p_prior
                col_b[t] = x_est

            # 2. Extract Column C
            col_c = col_a - col_b

            # Multiplier configuration constants
            k_fast = 2.0 / (float(fast_len) + 1.0)
            k_slow = 2.0 / (float(slow_len) + 1.0)
            k_sig = 2.0 / (float(signal_len) + 1.0)

            # 3. STRICT BAR 20 ACTIVATION ENGINE LOOP
            fast_ema = 0.0
            slow_ema = 0.0
            has_initialized =
