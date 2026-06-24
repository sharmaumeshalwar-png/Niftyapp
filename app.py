import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SCIENTIFIC UI THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AGCA Decoder V2", layout="wide")

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
            border: 1px solid #f43f5e;
            box-shadow: 0 4px 20px rgba(244, 63, 94, 0.15);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="decoder-block">
        <h1>🌌 AGCA Escape Engine (Nifty Bees Volume Framework)</h1>
        <p><b>Computational Anchor:</b> 01 Jan 2025 into 2026 | <b>Volume Source:</b> NIFTYBEES.NS Live Transactional Stream<br>
        Column D isolates the 10% structural escape breakouts from the 90% mean-reverting waves.</p>
    </div>
""", unsafe_allow_html=True)

if 'aet_v2_database' not in st.session_state:
    st.session_state.aet_v2_database = pd.DataFrame()

st.sidebar.subheader("🔬 Controls")
run_sync = st.sidebar.button("🔄 Execute Nifty & Bees Handshake")

# ==============================================================================
# 2. DATA PROCESSING MATRIX (DUAL ASSET COUPLING)
# ==============================================================================
if len(st.session_state.aet_v2_database) == 0 or run_sync:
    with st.spinner("Syncing Nifty Price & Nifty Bees Volume Arrays..."):
        try:
            # Pulling Nifty Index for accurate Close pricing
            nifty_raw = yf.download(tickers="^NSEI", start="2025-01-01", interval="1h", progress=False)
            # Pulling Nifty Bees for accurate transactional volume analysis
            bees_raw = yf.download(tickers="NIFTYBEES.NS", start="2025-01-01", interval="1h", progress=False)
            
            if not nifty_raw.empty and not bees_raw.empty:
                # Multi-index flattening if required
                if isinstance(nifty_raw.columns, pd.MultiIndex):
                    nifty_raw.columns = [col[0] for col in nifty_raw.columns]
                if isinstance(bees_raw.columns, pd.MultiIndex):
                    bees_raw.columns = [col[0] for col in bees_raw.columns]
                
                nifty_raw.columns = [str(col).strip().title() for col in nifty_raw.columns]
                bees_raw.columns = [str(col).strip().title() for col in bees_raw.columns]
                
                nifty_df = nifty_raw[['Close']].dropna()
                bees_df = bees_raw[['Volume']].dropna()
                
                # Combine datasets based on precise timestamp matching
                combined = nifty_df.join(bees_df, how='inner').sort_index()
                
                # Exclude active incomplete hour
                frozen_data = combined.iloc[:-1].copy() if len(combined) > 1 else combined.copy()
                frozen_data = frozen_data.reset_index()
                
                time_key = 'Datetime' if 'Datetime' in frozen_data.columns else frozen_data.columns[0]
                total_elements = len(frozen_data)
                
                col_a = np.array(frozen_data['Close'].values, dtype=float).flatten()
                col_v = np.array(frozen_data['Volume'].values, dtype=float).flatten()
                col_b = np.zeros(total_elements, dtype=float)
                col_c = np.zeros(total_elements, dtype=float)
                
                # Column D State Storage Array
                col_d_state = []
                
                if total_elements > 0:
                    col_b[0] = float(col_a[0])
                    col_c[0] = 0.0
                    col_d_state.append("Warmup")
                
                multiplier = 0.0001
                lookback = 20
                current_regime = "90% Mode"  # Initial base baseline
                
                # Core Engine Processing
                for t in range(1, total_elements):
                    current_a = float(col_a[t])
                    prev_b = float(col_b[t-1])
                    
                    # Mathematical Formulas
                    col_b[t] = prev_b + (multiplier * (current_a - prev_b))
                    col_c[t] = current_a - col_b[t]
                    
                    price_delta = current_a - float(col_a[t-1])
                    bees_volume = float(col_v[t]) if float(col_v[t]) > 0 else 1.0
                    
                    # Kinetic Tracking via Nifty Bees Volume Log
                    kinetic_energy = (price_delta ** 2) * np.log1p(bees_volume)
                    sign_flip = np.sign(col_c[t]) != np.sign(col_c[t-1])
                    
                    if t >= lookback:
                        # Extract rolling historical energy levels
                        recent_energy = []
                        for i in range(t-lookback, t):
                            p_d = col_a[i] - col_a[i-1]
                            v_m = col_v[i] if col_v[i] > 0 else 1.0
                            recent_energy.append((p_d ** 2) * np.log1p(v_m))
                        
                        energy_threshold = np.mean(recent_energy) * 1.5
                        
                        # Evaluating Mystery Breakpoints on Sign Changes
                        if sign_flip:
                            if kinetic_energy > energy_threshold:
                                current_regime = "10% Mode"  # Escape Track
                            else:
                                current_regime = "90% Mode"  # Reversion Track
                        
                        col_d_state.append(current_regime)
                    else:
                        col_d_state.append("Warmup")
                
                st.session_state.aet_v2_database = pd.DataFrame({
                    'Date_Time': [pd.to_datetime(x).strftime('%d %b %Y %H:%M') for x in frozen_data[time_key].values],
                    'Column A (Nifty Close)': [float(x) for x in col_a],
                    'Column B (Anchor)': [float(x) for x in col_b],
                    'Column C (Delta Variance)': [float(x) for x in col_c],
                    'Column D (Regime State)': col_d_state,
                    'NiftyBees Real Vol': [float(x) for x in col_v]
                })
                
        except Exception as ex:
            st.error(f"Matrix Synchronization Refused: {str(ex)}")

# ==============================================================================
# 3. STREAM PRESENTATION LAYER
# ==============================================================================
output_matrix = st.session_state.aet_v2_database.copy()

if not output_matrix.empty:
    st.write(f"### 📊 Processed Core Stream: **{len(output_matrix)} Data Intervals Verified**")
    inverted_view = output_matrix.iloc[::-1].reset_index(drop=True)
    
    def style_column_d(val):
        if "10%" in str(val):
            return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold; border: 1px solid #ef4444;'
        elif "90%" in str(val):
            return 'background-color: #14532d; color: #a7f3d0;'
        return ''

    styled_df = inverted_view.style.format({
        'Column A (Nifty Close)': '{:.2f}',
        'Column B (Anchor)': '{:.4f}',
        'Column C (Delta Variance)': '{:.4f}',
        'NiftyBees Real Vol': '{:,.0f}'
    }).map(style_column_d, subset=['Column D (Regime State)'])
    
    st.dataframe(styled_df, use_container_width=True)
else:
    st.info("System storage offline. Click 'Execute Nifty & Bees Handshake' control switch.")
