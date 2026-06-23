import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SCIENTIFIC UI THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AGCA-Filter 2026 Horizon", layout="wide")

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
        <h1>🌌 The AGCA-Filter Discovery Engine (2026 Target Mode)</h1>
        <p><b>Computational Anchor Point:</b> 01 January 2026 | <b>Structural Frame:</b> Active 2026 Micro-Timeline<br>
        System anchors firmly on the first trading candle of 2026. All subsequent rows freeze dynamically step-by-step from that node.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE MATRIX REGISTRY
# ==============================================================================
if 'agca_2026_target_database' not in st.session_state:
    st.session_state.agca_2026_target_database = pd.DataFrame()

# Control Dashboard Layout
st.sidebar.subheader("🔬 2026 Target Engine Controls")
run_sync = st.sidebar.button("🔄 Execute 2026 Handshake Loop")
reset_system = st.sidebar.button("🗑️ Reset 2026 Storage Core")

if reset_system:
    st.session_state.agca_2026_target_database = pd.DataFrame()
    st.sidebar.success("2026 Dataset wiped successfully.")
    st.rerun()

# ==============================================================================
# 3. ADVANCED SCIENTIFIC DATA PROCESSING MATRIX (ANCHOR: 2026)
# ==============================================================================
if len(st.session_state.agca_2026_target_database) == 0 or run_sync:
    with st.spinner("Compiling High-Fidelity 2026 Target Matrix..."):
        try:
            # Strictly bounding data pull from 01 Jan 2026 onwards
            raw_feed = yf.download(tickers="^NSEI", start="2026-01-01", interval="1h", progress=False)
            
            if not raw_feed.empty:
                if isinstance(raw_feed.columns, pd.MultiIndex):
                    raw_feed.columns = [col[0] for col in raw_feed.columns]
                raw_feed.columns = [str(col).strip().title() for col in raw_feed.columns]
                
                raw_feed = raw_feed.dropna(subset=['Close']).sort_index(ascending=True)
                
                # Exclude the unstable running live hourly candle to lock computational tracks
                if len(raw_feed) > 1:
                    frozen_candles = raw_feed.iloc[:-1].copy()
                else:
                    frozen_candles = raw_feed.copy()
                
                frozen_candles = frozen_candles.reset_index()
                time_key = 'Datetime' if 'Datetime' in frozen_candles.columns else frozen_candles.columns[0]
                
                total_elements = len(frozen_candles)
                
                # Initialize structured empty vector arrays safely
                col_a = np.array(frozen_candles['Close'].values, dtype=float).flatten()
                col_b = np.zeros(total_elements, dtype=float)
                col_c = np.zeros(total_elements, dtype=float)
                col_d = np.zeros(total_elements, dtype=float)
                col_e = np.zeros(total_elements, dtype=float)
                col_f = np.zeros(total_elements, dtype=float)
                col_g = np.zeros(total_elements, dtype=float)
                dynamic_alpha = np.zeros(total_elements, dtype=float)
                time_list = []
                
                # Strict 01 Jan 2026 Seed Hard-Locking ($B_0 = A_0$)
                if total_elements > 0:
                    col_b[0] = float(col_a[0])
                    col_c[0] = 0.0
                    col_d[0] = 0.0
                    col_e[0] = 0.0
                    col_f[0] = 0.0
                    col_g[0] = 0.0
                    dynamic_alpha[0] = 0.0001
                    time_list.append(pd.to_datetime(frozen_candles[time_key].values[0]).strftime('%d %b %Y %H:%M'))
                
                # Core Scientific Discovery Loop Processing
                for t in range(1, total_elements):
                    # Format dynamic timestamp tracking list arrays
                    time_list.append(pd.to_datetime(frozen_candles[time_key].values[t]).strftime('%d %b %Y %H:%M'))
                    
                    # 1. Compute Information Distance Metric safely using raw scalars
                    current_a = float(col
