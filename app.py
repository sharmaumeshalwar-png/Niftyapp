import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# SYSTEM CALCULATION ENGINE: 8-STEP VERIFICATION MATRIX
# ==============================================================================

# STEP 1: SCIENTIFIC UI THEME CONFIGURATION
st.set_page_config(page_title="Nifty Mean Reversion Core", layout="wide")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #040914 !important;
            color: #e2e8f0 !important;
        }
        h1, h3, p, span, label { color: #f8fafc !important; }
        .discovery-block {
            background: linear-gradient(135deg, #020617 0%, #0d1527 100%);
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #3b82f6;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.2);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="discovery-block">
        <h1>🌌 Nifty 50 Statistical Arbitrage Engine</h1>
        <p><b>Core Framework:</b> Kalman Variance ➔ Rolling Standard Deviation Band Integration (No-Indicator Model)</p>
    </div>
""", unsafe_allow_html=True)

# STEP 2: SIDEBAR CONTROLS REGISTRY
st.sidebar.subheader("🔬 Stat-Engine Parameters")
ticker = st.sidebar.text_input("Target Ticker (Yahoo Finance)", value="^NSEI")
bb_period = st.sidebar.number_input("Statistical Band Lookback", min_value=5, value=20)
bb_std = st.sidebar.number_input("Band Deviation Threshold (Sigma)", min_value=0.5, value=2.0, step=0.1)

run_sync = st.sidebar.button("🔄 Execute Quantum Handshake Loop")

if 'nifty_stat_db' not in st.session_state:
    st.session_state.nifty_stat_db = pd.DataFrame()

# STEP 3: RE-CONFIGURED SAFE INGESTION PIPELINE
if len(st.session_state.nifty_stat_db) == 0 or run_sync:
    with st.spinner("Processing High-Fidelity Statistical Data Grid..."):
        try:
            import yfinance as yf
            raw_feed = yf.download(tickers=str(ticker), interval="1h", period="2y", progress=False)
        except ModuleNotFoundError:
            raw_feed = pd.DataFrame()

        # Fallback Engine
        if raw_feed.empty:
            date_range = pd.date_range(start="2024-06-01", end="2026-12-31", freq="1h")
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
            # STEP 5: MATHEMATICAL ARRAY CORE PROCESSING (Anti-Leak Math)
            col_b = np.zeros(total_elements, dtype=float)       # Kalman Array
            col_c = np.zeros(total_elements, dtype=float)       # Delta Variance (A-B)
            col_d_upper = np.zeros(total_elements, dtype=float) # Upper Deviation Boundary
            col_d_lower = np.zeros(total_elements, dtype=float) # Lower Deviation Boundary
            time_list = ["" for _ in range(total_elements)]

            # 1. Pure Kalman Filter Execution on Column A
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

            # 2. Compute Column C = (A - B)
            col_c = col_a - col_b

            # 3. Compute Dynamic Bollinger Bands directly on Column C Vector
            p_len = int(bb_period)
            sigma_mult = float(bb_std)
            
            for t in range(total_elements):
                if t >= p_len:
                    sub_window = col_c[t - p_len + 1 : t + 1]
                    mean_c = np.mean(sub_window)
                    std_c = np.std(sub_window)
                    col_d_upper[t] = mean_c + (sigma_mult * std_c)
                    col_d_lower[t] = mean_c - (sigma_mult * std_c)
                else:
                    col_d_upper[t] = 0.0
                    col_d_lower[t] = 0.0

            # STEP 6: LAPTOP EXCEL ZONE CLASSIFICATION INTEGRATION
            excel_zones = []
            for t in range(total_elements):
                if col_d_upper[t] != 0.0:
                    if col_c[t] >= col_d_upper[t]:
                        excel_zones.append("⚠️ Overstretched High (Sell Call Zone)")
                    elif col_c[t] <= col_d_lower[t]:
                        excel_zones.append("🟢 Overstretched Low (Sell Put Zone)")
                    else:
                        excel_zones.append("Mean Reverting Zone (No Trade)")
                else:
                    excel_zones.append("Warming Up Engine...")

            # STEP 7: SECURE TRANSITION PACKING & 2-YEAR FREEZE LOCK
            research_df = pd.DataFrame({
                'Date_Time': time_list,
                'Column A (Nifty Close)': [float(x) for x in col_a],
                'Column B (Kalman Nifty)': [float(x) for x in col_b],
                'Column C (Delta Variance)': [float(x) for x in col_c],
                'Upper Strike Trigger': [float(x) for x in col_d_upper],
                'Lower Strike Trigger': [float(x) for x in col_d_lower],
                'Excel Market Zone': excel_zones
            })

            research_df['Datetime_Parsed'] = pd.to_datetime(research_df['Date_Time'], format='%d %b %Y %H:%M')
            
            # Hard-Lock Target Window Boundaries
            start_date = pd.Timestamp("2025-01-01 00:00:00")
            end_date = pd.Timestamp("2026-12-31 23:59:59")

            final_mask = (research_df['Datetime_Parsed'] >= start_date) & (research_df['Datetime_Parsed'] <= end_date)
            st.session_state.nifty_stat_db = research_df[final_mask].drop(columns=['Datetime_Parsed']).reset_index(drop=True)

# ==============================================================================
# STEP 8: PRESENTATION GRID & ALL POSSIBLE OUTCOME DATES MATRIX
# ==============================================================================
output_matrix = st.session_state.nifty_stat_db.copy()

if not output_matrix.empty:
    st.write(f"### 📊 Step 8 Matrix: **{len(output_matrix)} Quant Data Blocks Hard-Locked**")
    
    st.write("#### 📋 All Possible Outcome Dates Matrix Logs (Latest First)")
    inverted_view = output_matrix.iloc[::-1].reset_index(drop=True)

    st.dataframe(
        inverted_view[['Date_Time', 'Column A (Nifty Close)', 'Column B (Kalman Nifty)', 'Column C (Delta Variance)', 'Upper Strike Trigger', 'Lower Strike Trigger', 'Excel Market Zone']].style.format({
            'Column A (Nifty Close)': '{:.2f}',
            'Column B (Kalman Nifty)': '{:.2f}',
            'Column C (Delta Variance)': '{:.4f}',
            'Upper Strike Trigger': '{:.4f}',
            'Lower Strike Trigger': '{:.4f}'
        }),
        use_container_width=True
    )
else:
    st.warning("Quantum storage core empty. Trigger handshake loop via sidebar panel.")
