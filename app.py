import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SCIENTIFIC UI THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AGCA-Filter Scientific Discovery", layout="wide")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #050b14 !important;
            color: #e2e8f0 !important;
        }
        h1, h3, p, span, label { color: #f8fafc !important; }
        .discovery-block {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
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
        <h1>🌌 The Adaptive Geometric Cascade Attractor (AGCA-Filter)</h1>
        <p><b>Scientific Classification:</b> Non-Stationary Signal Processing Engine<br>
        <b>Core Mechanics:</b> Multiplier is NOT fixed. It uses local Fisher Information-like scalar weights to self-tune matrix cascade bounds sequentially starting from June 2025.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE STATE-MACHINE STORAGE
# ==============================================================================
if 'agca_database' not in st.session_state:
    st.session_state.agca_database = pd.DataFrame()

# Sidebar automation switches
st.sidebar.subheader("🔬 Discovery Control Panel")
trigger_handshake = st.sidebar.button("🔄 Execute Space-Grade Sync Loop")
wipe_system = st.sidebar.button("🗑️ Hard Reset Research Matrix")

if wipe_system:
    st.session_state.agca_database = pd.DataFrame()
    st.sidebar.success("Research matrix storage cleared.")
    st.rerun()

# ==============================================================================
# 3. ADVANCED SCIENTIFIC DATA PROCESSING MATRIX
# ==============================================================================
if len(st.session_state.agca_database) == 0 or trigger_handshake:
    with st.spinner("Processing High-Fidelity Geometric Arrays..."):
        try:
            # Download baseline from anchored timeline
            raw_feed = yf.download(tickers="^NSEI", start="2025-06-01", interval="1h", progress=False)
            
            if not raw_feed.empty:
                if isinstance(raw_feed.columns, pd.MultiIndex):
                    raw_feed.columns = [col[0] for col in raw_feed.columns]
                raw_feed.columns = [str(col).strip().title() for col in raw_feed.columns]
                
                raw_feed = raw_feed.dropna(subset=['Close']).sort_index(ascending=True)
                
                # Drop dynamic open ongoing candle to freeze previous timeline
                if len(raw_feed) > 1:
                    frozen_candles = raw_feed.iloc[:-1].copy()
                else:
                    frozen_candles = raw_feed.copy()
                
                frozen_candles = frozen_candles.reset_index()
                time_key = 'Datetime' if 'Datetime' in frozen_candles.columns else frozen_candles.columns[0]
                
                total_elements = len(frozen_candles)
                
                # Pre-allocate mathematical vector arrays
                col_a = frozen_candles['Close'].astype(float).values
                col_b = np.zeros(total_elements, dtype=float)
                col_c = np.zeros(total_elements, dtype=float)
                col_d = np.zeros(total_elements, dtype=float)
                col_e = np.zeros(total_elements, dtype=float)
                col_f = np.zeros(total_elements, dtype=float)
                col_g = np.zeros(total_elements, dtype=float)
                dynamic_alpha = np.zeros(total_elements, dtype=float)
                
                # Base Seed Hard-Locking ($B_0 = A_0$)
                if total_elements > 0:
                    col_b[0] = col_a[0]
                    col_c[0] = 0.0
                    col_d[0] = 0.0
                    col_e[0] = 0.0
                    col_f[0] = 0.0
                    col_g[0] = 0.0
                    dynamic_alpha[0] = 0.0001
                
                # Core Scientific Discovery Loop Processing
                for t in range(1, total_elements):
                    # 1. Compute Information Distance Metric (Fisher Scalar approximation)
                    # It measures how fast raw data is escaping the smooth manifold
                    distance_metric = abs(col_a[t] - col_b[t-1]) / (col_b[t-1] if col_b[t-1] != 0 else 1)
                    
                    # 2. Adaptive Weight Allocation via Sigmoid scaling bounded safely
                    # Multiplier scales dynamically between 0.00005 (slow market) and 0.0025 (fast blast market)
                    alpha_t = 0.00005 + (0.0025 / (1.0 + np.exp(-1000.0 * (distance_metric - 0.002))))
                    dynamic_alpha[t] = alpha_t
                    
                    # 3. Cascaded Pipeline Synthesis using the adaptive alpha token
                    col_b[t] = col_b[t-1] + (alpha_t * (col_a[t] - col_b[t-1]))
                    col_c[t] = col_a[t] - col_b[t]
                    
                    col_d[t] = col_d[t-1] + (alpha_t * (col_c[t] - col_d[t-1]))
                    col_e[t] = col_e[t-1] + (alpha_t * (col_d[t] - col_e[t-1]))
                    col_f[t] = col_f[t-1] + (alpha_t * (col_e[t] - col_f[t-1]))
                    
                    # Fluid kinematic momentum tracking step
                    col_g[t] = col_f[t] - col_f[t-1]
                
                # Assembly Matrix Construction
                research_df = pd.DataFrame({
                    'Date_Time': pd.to_datetime(frozen_candles[time_key]).dt.strftime('%d %b %Y %H:%M'),
                    'Column A (Raw Close)': col_a,
                    'Adaptive Alpha (α)': dynamic_alpha,
                    'Column B (Smooth Node)': col_b,
                    'Column C (Delta Variance)': col_c,
                    'Column D (Exp Tier 1)': col_d,
                    'Column E (Exp Tier 2)': col_e,
                    'Column F (Exp Tier 3)': col_f,
                    'Column G (AGCA Attractor Signal)': col_g
                })
                
                st.session_state.agca_database = research_df
                
        except Exception as ex:
            st.error(f"Scientific array generation pipeline compromised: {str(ex)}")

# ==============================================================================
# 4. SCIENTIFIC PRESENTATION GRID DISPLAY
# ==============================================================================
output_matrix = st.session_state.agca_database.copy()

if not output_matrix.empty:
    st.write(f"### 📊 Processed High-Fidelity Signal Arrays: **{len(output_matrix)} Data Blocks Locked**")
    
    # Invert view so the absolute latest mathematical discovery sits at row 1
    inverted_view = output_matrix.iloc[::-1].reset_index(drop=True)
    
    st.dataframe(
        inverted_view.style.format({
            'Column A (Raw Close)': '{:.2f}',
            'Adaptive Alpha (α)': '{:.6f}',
            'Column B (Smooth Node)': '{:.4f}',
            'Column C (Delta Variance)': '{:.4f}',
            'Column D (Exp Tier 1)': '{:.4f}',
            'Column E (Exp Tier 2)': '{:.4f}',
            'Column F (Exp Tier 3)': '{:.4f}',
            'Column G (AGCA Attractor Signal)': '{:.6f}'
        }),
        use_container_width=True
    )
else:
    st.warning("Quantum storage core empty. Trigger pipeline loop via sidebar panel.")
