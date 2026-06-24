import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SCIENTIFIC UI THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AGCA 2025 Historical Core", layout="wide")

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
            border: 1px solid #ef4444;
            box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="decoder-block">
        <h1>🌌 AGCA Historical Freeze Engine — 2025 Baseline Locked</h1>
        <p><b>Computational Anchor:</b> Strict 01 Jan 2025 Point-to-Point Freeze | <b>Volume Source:</b> NIFTYBEES.NS Daily Volume Mass<br>
        Column D locks exclusive system signals: <b>10% Mode</b> (Escape Velocity) or <b>90% Mode</b> (Attractor Reversion).</p>
    </div>
""", unsafe_allow_html=True)

if 'agca_2025_freeze_db' not in st.session_state:
    st.session_state.agca_2025_freeze_db = pd.DataFrame()

st.sidebar.subheader("🔬 Engine Controls")
run_sync = st.sidebar.button("🔄 Initialize 2025 Historical Handshake")

# ==============================================================================
# 2. COMPUTATIONAL HISTORICAL FREEZE PIPELINE (ANCHOR: 2025)
# ==============================================================================
if len(st.session_state.agca_2025_freeze_db) == 0 or run_sync:
    with st.spinner("Fetching and flattening 2025-2026 data loops..."):
        try:
            # Strictly anchoring from 01 Jan 2025 (Daily timeframe supports unlimited lookback)
            nifty_raw = yf.download(tickers="^NSEI", start="2025-01-01", interval="1d", progress=False)
            bees_raw = yf.download(tickers="NIFTYBEES.NS", start="2025-01-01", interval="1d", progress=False)
            
            if not nifty_raw.empty and not bees_raw.empty:
                # Multi-index headers protection
                if isinstance(nifty_raw.columns, pd.MultiIndex):
                    nifty_raw.columns = [col[0] for col in nifty_raw.columns]
                if isinstance(bees_raw.columns, pd.MultiIndex):
                    bees_raw.columns = [col[0] for col in bees_raw.columns]
                
                nifty_raw.columns = [str(col).strip().title() for col in nifty_raw.columns]
                bees_raw.columns = [str(col).strip().title() for col in bees_raw.columns]
                
                # Resilient mapping framework (0% row drop guarantee)
                nifty_df = nifty_raw[['Close']].dropna()
                nifty_df['Real_Vol'] = nifty_df.index.map(bees_raw['Volume'])
                nifty_df['Real_Vol'] = nifty_df['Real_Vol'].fillna(bees_raw['Volume'].median())
                
                nifty_df = nifty_df.sort_index()
                
                # Freeze current active running candle
                frozen_data = nifty_df.iloc[:-1].copy() if len(nifty_df) > 1 else nifty_df.copy()
                frozen_data = frozen_data.reset_index()
                
                time_key = 'Date' if 'Date' in frozen_data.columns else frozen_data.columns[0]
                total_elements = len(frozen_data)
                
                col_a = np.array(frozen_data['Close'].values, dtype=float).flatten()
                col_v = np.array(frozen_data['Real_Vol'].values, dtype=float).flatten()
                col_b = np.zeros(total_elements, dtype=float)
                col_c = np.zeros(total_elements, dtype=float)
                
                col_d_state = []
                
                # Step 1: Strict Hard-Lock Node Seeding
                if total_elements > 0:
                    col_b[0] = float(col_a[0])
                    col_c[0] = 0.0
                    col_d_state.append("Warmup")
                
                multiplier = 0.0001
                lookback = 10  # Optimized window for daily velocity tracks
                current_regime = "90% Mode"
                
                # Step 2 to Step 8 Iterative Mathematical Loop
                for t in range(1, total_elements):
                    current_a = float(col_a[t])
                    prev_b = float(col_b[t-1])
                    
                    # Target Rule Formula Matrix
                    col_b[t] = prev_b + (multiplier * (current_a - prev_b))
                    col_c[t] = current_a - col_b[t]
                    
                    price_delta = current_a - float(col_a[t-1])
                    bees_volume = float(col_v[t]) if float(col_v[t]) > 0 else 1.0
                    
                    # Kinetic Energy Formulation (Coupling Price with True Volume Mass)
                    kinetic_energy = (price_delta ** 2) * np.log1p(bees_volume)
                    sign_flip = np.sign(col_c[t]) != np.sign(col_c[t-1])
                    
                    if t >= lookback:
                        recent_energy = []
                        for i in range(t-lookback, t):
                            p_d = col_a[i] - col_a[i-1]
                            v_m = col_v[i] if col_v[i] > 0 else 1.0
                            recent_energy.append((p_d ** 2) * np.log1p(v_m))
                        
                        # Energy benchmark threshold validation
                        energy_threshold = np.mean(recent_energy) * 1.4
                        
                        # Sign Flip Mystery Core Evaluation
                        if sign_flip:
                            if kinetic_energy > energy_threshold:
                                current_regime = "10% Mode"
                            else:
                                current_regime = "90% Mode"
                        
                        col_d_state.append(current_regime)
                    else:
                        col_d_state.append("Warmup")
                
                # Store structural variables into session state database
                st.session_state.agca_2025_freeze_db = pd.DataFrame({
                    'Date': [pd.to_datetime(x).strftime('%d %b %Y') for x in frozen_data[time_key].values],
                    'Column A (Nifty Close)': [float(x) for x in col_a],
                    'Column B (Anchor)': [float(x) for x in col_b],
                    'Column C (Delta Variance)': [float(x) for x in col_c],
                    'Column D (Regime State)': col_d_state,
                    'NiftyBees Vol': [float(x) for x in col_v]
                })
                
                st.sidebar.success("Database Frozen Successfully from 2025.")
            else:
                st.error("Assets data pull rejected by server.")
                
        except Exception as ex:
            st.error(f"Core execution breakdown: {str(ex)}")

# ==============================================================================
# 3. PRESENTATION GRID LAYER (LATEST DATA ON TOP)
# ==============================================================================
output_matrix = st.session_state.agca_2025_freeze_db.copy()

if not output_matrix.empty:
    st.write(f"### 📊 Processed Core Stream: **{len(output_matrix)} Trading Days Locked**")
    inverted_view = output_matrix.iloc[::-1].reset_index(drop=True)
    
    def style_column_d(val):
        if "10%" in str(val):
            return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold; border: 1px solid #ef4444;'
        elif "90%" in
