import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SCIENTIFIC UI THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AGCA-Filter 2026 Stable", layout="wide")

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
        <h1>🌌 The AGCA-Filter Discovery Engine (Line 138 Absolute Fix)</h1>
        <p><b>Anchor Point:</b> 01 January 2026 | <b>Structural Patch:</b> Explicit 1-D sequence alignment forced on dataframe assembly parameters to resolve array dimension matrix errors completely.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE MATRIX REGISTRY
# ==============================================================================
if 'agca_2026_database' not in st.session_state:
    st.session_state.agca_2026_database = pd.DataFrame()

# Control Dashboard Layout
st.sidebar.subheader("🔬 2026 Engine Controls")
run_sync = st.sidebar.button("🔄 Execute 2026 Handshake Loop")
reset_system = st.sidebar.button("🗑️ Reset 2026 Storage Core")

if reset_system:
    st.session_state.agca_2026_database = pd.DataFrame()
    st.sidebar.success("2026 Dataset wiped successfully.")
    st.rerun()

# ==============================================================================
# 3. ADVANCED SCIENTIFIC DATA PROCESSING MATRIX (FIXED ENGINE)
# ==============================================================================
if len(st.session_state.agca_2026_database) == 0 or run_sync:
    with st.spinner("Compiling High-Fidelity 2026 Matrix..."):
        try:
            # Bounding data pull from 01 Jan 2026 onwards
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
                    current_a = float(col_a[t])
                    prev_b = float(col_b[t-1])
                    
                    distance_metric = abs(current_a - prev_b) / (prev_b if prev_b != 0 else 1.0)
                    
                    # 2. Safe scalar casting within sigmoid loop to prevent array dimensions error
                    exponent_value = -1000.0 * (float(distance_metric) - 0.002)
                    exponent_value = max(-50.0, min(50.0, exponent_value))
                    
                    alpha_t = 0.00005 + (0.0025 / (1.0 + np.exp(exponent_value)))
                    dynamic_alpha[t] = alpha_t
                    
                    # 3. Cascaded Pipeline Matrix Formulations
                    col_b[t] = prev_b + (alpha_t * (current_a - prev_b))
                    col_c[t] = current_a - col_b[t]
                    
                    col_d[t] = float(col_d[t-1]) + (alpha_t * (col_c[t] - float(col_d[t-1])))
                    col_e[t] = float(col_e[t-1]) + (alpha_t * (col_d[t] - float(col_e[t-1])))
                    col_f[t] = float(col_f[t-1]) + (alpha_t * (col_e[t] - float(col_f[t-1])))
                    
                    # Final Velocity Tracker Generation
                    col_g[t] = col_f[t] - col_f[t-1]
                
                # [LINE 138 FIX PROTECTION] Converting everything explicitly to 1D flattened lists to ensure dataframe sync safety
                research_df = pd.DataFrame({
                    'Date_Time': list(time_list),
                    'Column A (Raw Close)': [float(x) for x in col_a],
                    'Adaptive Alpha (α)': [float(x) for x in dynamic_alpha],
                    'Column B (Smooth Node)': [float(x) for x in col_b],
                    'Column C (Delta Variance)': [float(x) for x in col_c],
                    'Column D (Exp Tier 1)': [float(x) for x in col_d],
                    'Column E (Exp Tier 2)': [float(x) for x in col_e],
                    'Column F (Exp Tier 3)': [float(x) for x in col_f],
                    'Column G (AGCA Attractor Signal)': [float(x) for x in col_g]
                })
                
                st.session_state.agca_2026_database = research_df
                
        except Exception as ex:
            st.error(f"Scientific array generation pipeline compromised: {str(ex)}")

# ==============================================================================
# 4. PRESENTATION GRID DISPLAY
# ==============================================================================
output_matrix = st.session_state.agca_2026_database.copy()

if not output_matrix.empty:
    st.write(f"### 📊 2026 Micro-Matrix Stream: **{len(output_matrix)} Data Blocks Hard-Locked**")
    
    # Invert view layers so that the absolute latest completed hour sits neatly on top
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
    st.warning("Quantum storage core empty for year 2026. Trigger handshake loop via sidebar panel.")
