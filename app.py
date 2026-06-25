import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# SYSTEM CALCULATION ENGINE: 8-STEP VERIFICATION MATRIX
# ==============================================================================

# STEP 1: SCIENTIFIC UI THEME CONFIGURATION
st.set_page_config(page_title="Coppock-Engine 2026 Perfect", layout="wide")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #050b14 !important;
            color: #e2e8f0 !important;
        }
        h1, h3, p, span, label { color: #f8fafc !important; }
        .discovery-block {
            background: linear-gradient(135deg, #020617 0%, #0f172a 100%);
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #10b981;
            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="discovery-block">
        <h1>🌌 The Coppock & Column C Discovery Engine (Final Stable Fix)</h1>
        <p><b>Computational Core:</b> Dual-Vector Convergence | <b>Window Lock:</b> Jan 2025 - Dec 2026 (1-Hour Grid Sync)</p>
    </div>
""", unsafe_allow_html=True)

# STEP 2: SESSION STATE MATRIX REGISTRY
if 'coppock_2026_perfect_db' not in st.session_state:
    st.session_state.coppock_2026_perfect_db = pd.DataFrame()

st.sidebar.subheader("🔬 Step 2: Precision Controls")
ticker = st.sidebar.text_input("Target Ticker (Yahoo Finance)", value="BTC-USD")
delta_lookback = st.sidebar.number_input("Delta Lookback (Column C)", min_value=1, value=14)
long_roc = st.sidebar.number_input("Coppock Long ROC", min_value=1, value=14)
short_roc = st.sidebar.number_input("Coppock Short ROC", min_value=1, value=11)
wma_smoothing = st.sidebar.number_input("Coppock WMA Smoothing", min_value=1, value=10)

run_sync = st.sidebar.button("🔄 Execute 2026 Handshake Loop")
reset_system = st.sidebar.button("🗑️ Reset 2026 Storage Core")

if reset_system:
    st.session_state.coppock_2026_perfect_db = pd.DataFrame()
    st.sidebar.success("2026 Engine Dataset wiped successfully.")
    st.rerun()

# STEP 3: RE-CONFIGURED SAFE INGESTION PIPELINE
if len(st.session_state.coppock_2026_perfect_db) == 0 or run_sync:
    with st.spinner("Compiling High-Fidelity Coppock Target Matrix..."):
        try:
            # Safe Local Execution Injection to completely isolate multi-thread errors
            import yfinance as yf
            
            raw_feed = yf.download(tickers=str(ticker), interval="1h", period="2y", progress=False)
            using_fallback = False
            
            # Automated Fallback Execution Block if API fails or blocks
            if raw_feed is None or raw_feed.empty:
                using_fallback = True
                date_range = pd.date_range(start="2024-06-01", end="2026-12-31", freq="1h")
                np.random.seed(42)
                simulated_close = 60000 + np.cumsum(np.random.normal(0, 150, len(date_range)))
                raw_feed = pd.DataFrame({
                    'Open': simulated_close - 50, 'High': simulated_close + 100,
                    'Low': simulated_close - 100, 'Close': simulated_close, 'Volume': 500
                }, index=date_range)
                raw_feed.index.name = 'Datetime'

            # STEP 4: VECTOR BASELINE & ARRAY ANCHOR SETUP
            # Extractions directly from values to prevent MultiIndex or single column name mismatches
            if isinstance(raw_feed.columns, pd.MultiIndex):
                raw_close_array = raw_feed['Close'].values.flatten()
                raw_open_array = raw_feed['Open'].values.flatten()
                raw_high_array = raw_feed['High'].values.flatten()
                raw_low_array = raw_feed['Low'].values.flatten()
            else:
                raw_close_array = raw_feed.iloc[:, raw_feed.columns.get_loc('Close') if 'Close' in raw_feed.columns else 3].values.flatten()
                raw_open_array = raw_feed.iloc[:, raw_feed.columns.get_loc('Open') if 'Open' in raw_feed.columns else 0].values.flatten()
                raw_high_array = raw_feed.iloc[:, raw_feed.columns.get_loc('High') if 'High' in raw_feed.columns else 1].values.flatten()
                raw_low_array = raw_feed.iloc[:, raw_feed.columns.get_loc('Low') if 'Low' in raw_feed.columns else 2].values.flatten()

            raw_dates = raw_feed.index.to_numpy()
            total_elements = len(raw_close_array)

            if total_elements > 0:
                # STEP 5: ROLLING ARRAY PROCESSING ENGINE (Anti-Leak Math)
                col_a = np.array(raw_close_array, dtype=float)
                col_c = np.zeros(total_elements, dtype=float)
                coppock_curve = np.zeros(total_elements, dtype=float)
                time_list = ["" for _ in range(total_elements)]

                # Pure Column C Vector Loop
                for t in range(total_elements):
                    time_list[t] = pd.to_datetime(raw_dates[t]).strftime('%d %b %Y %H:%M')
                    if t >= int(delta_lookback):
                        col_c[t] = col_a[t] - col_a[t - int(delta_lookback)]
                    else:
                        col_c[t] = 0.0

                # Pure Coppock Vector Arrays Setup
                roc_l = np.zeros(total_elements, dtype=float)
                roc_s = np.zeros(total_elements, dtype=float)
                for t in range(total_elements):
                    if t >= int(long_roc) and col_a[t - int(long_roc)] != 0:
                        roc_l[t] = ((col_a[t] - col_a[t - int(long_roc)]) / col_a[t - int(long_roc)]) * 100
                    if t >= int(short_roc) and col_a[t - int(short_roc)] != 0:
                        roc_s[t] = ((col_a[t] - col_a[t - int(short_roc)]) / col_a[t - int(short_roc)]) * 100

                matrix_sum = roc_l + roc_s

                # Pure WMA Linear Vector Math Loop
                w_len = int(wma_smoothing)
                weights = np.arange(1, w_len + 1)
                w_sum = weights.sum()
                for t in range(total_elements):
                    if t >= w_len - 1:
                        sub_segment = matrix_sum[t - w_len + 1 : t + 1]
                        coppock_curve[t] = np.dot(sub_segment, weights) / w_sum
                    else:
                        coppock_curve[t] = 0.0

                # STEP 6: LAPTOP EXCEL ZONE CLASSIFICATION INTEGRATION
                excel_zones = []
                for val in coppock_curve:
                    if val >= 2.5: excel_zones.append("⚠️ Extreme Overbought (+2.5)")
                    elif val <= -2.5: excel_zones.append("🟢 Extreme Oversold (-2.5)")
                    else: excel_zones.append("Normal Zone")

                # STEP 7: SECURE TRANSITION PACKING & 2-YEAR FREEZE LOCK
                research_df = pd.DataFrame({
                    'Date_Time': time_list,
                    'Column A (Raw Close)': [float(x) for x in col_a],
                    'Column C (Delta Variance)': [float(x) for x in col_c],
                    'Coppock Curve': [float(x) for x in coppock_curve],
                    'Excel Market Zone': excel_zones,
                    'Open_Raw': [float(x) for x in raw_open_array],
                    'High_Raw': [float(x) for x in raw_high_array],
                    'Low_Raw': [float(x) for x in raw_low_array]
                })

                research_df['Datetime_Parsed'] = pd.to_datetime(research_df['Date_Time'], format='%d %b %Y %H:%M')
                start_date = pd.Timestamp("2025-01-01 00:00:00")
                end_date = pd.
