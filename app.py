import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SCIENTIFIC UI THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AGCA 1H Absolute Engine", layout="wide")

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
            border: 1px solid #3b82f6;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="decoder-block">
        <h1>🌌 AGCA Escape Engine — Safe Mode 1H Core</h1>
        <p><b>Timeframe:</b> Rolling 59-Day 1-Hour Window | <b>Volume Mapping:</b> Resilient Dual-Asset Alignment<br>
        Column D Displays Clean State Output: <b>10% Mode</b> (Escape/Trend) or <b>90% Mode</b> (Reversion).</p>
    </div>
""", unsafe_allow_html=True)

if 'agca_robust_db' not in st.session_state:
    st.session_state.agca_robust_db = pd.DataFrame()

st.sidebar.subheader("🔬 Control Room")
run_sync = st.sidebar.button("🔄 Force Core 1H Handshake")

# ==============================================================================
# 2. SEAMLESS DUAL-ASSET SYNCHRONIZATION MATRIX
# ==============================================================================
if len(st.session_state.agca_robust_db) == 0 or run_sync:
    with st.spinner("Executing Data Stream Pipeline..."):
        try:
            # Safe boundary window formulation for hourly frames
            start_date = (datetime.now() - timedelta(days=55)).strftime('%Y-%m-%d')
            
            # Fetching Datasets
            nifty_df = yf.download(tickers="^NSEI", start=start_date, interval="1h", progress=False)
            bees_df = yf.download(tickers="NIFTYBEES.NS", start=start_date, interval="1h", progress=False)
            
            if nifty_df.empty:
                st.error("Yahoo Finance did not return NIFTY Index price stream.")
            else:
                # Multi-index column flattening protection
                if isinstance(nifty_df.columns, pd.MultiIndex):
                    nifty_df.columns = [col[0] for col in nifty_df.columns]
                if isinstance(bees_df.columns, pd.MultiIndex):
                    bees_df.columns = [col[0] for col in bees_df.columns]
                
                nifty_df.columns = [str(col).strip().title() for col in nifty_df.columns]
                
                # Resilient Volume Fallback Structure
                if not bees_df.empty:
                    bees_df.columns = [str(col).strip().title() for col in bees_df.columns]
                    # Map volume safely to Nifty index index to prevent data drop
                    nifty_df['Real_Vol'] = nifty_df.index.map(bees_df['Volume'])
                    # If any timestamp doesn't match, fill with median volume to keep system running
                    nifty_df['Real_Vol'] = nifty_df['Real_Vol'].fillna(bees_df['Volume'].median())
                else:
                    # Absolute emergency fallback if NiftyBees API completely timeouts
                    nifty_df['Real_Vol'] = 100000.0
                
                nifty_df = nifty_df.dropna(subset=['Close']).sort_index()
                
                # Freeze incomplete running candle
                frozen_data = nifty_df.iloc[:-1].copy() if len(nifty_df) > 1 else nifty_df.copy()
                frozen_data = frozen_data.reset_index()
                
                time_key = 'Datetime' if 'Datetime' in frozen_data.columns else frozen_data.columns[0]
                total_elements = len(frozen_data)
                
                col_a = np.array(frozen_data['Close'].values, dtype=float).flatten()
                col_v = np.array(frozen_data['Real_Vol'].values, dtype=float).flatten()
                col_b = np.zeros(total_elements, dtype=float)
                col_c = np.zeros(total_elements, dtype=float)
                
                col_d_state = []
                
                # Seeding step 1
                if total_elements > 0:
                    col_b[0] = float(col_a[0])
                    col_c[0] = 0.0
                    col_d_state.append("Warmup")
                
                multiplier = 0.0001
                lookback = 14
                current_regime = "90% Mode"
                
                # High-Speed Vector Iterator
                for t in range(1, total_elements):
                    current_a = float(col_a[t])
                    prev_b = float(col_b[t-1])
                    
                    # Core Equation Rule: b2 = b1 + 0.0001 * (a2 - b1)
                    col_b[t] = prev_b + (multiplier * (current_a - prev_b))
                    col_c[t] = current_a - col_b[t]
                    
                    price_delta = current_a - float(col_a[t-1])
                    current_vol = float(col_v[t]) if float(col_v[t]) > 0 else 1.0
                    
                    # Kinetic Energy Extraction
                    kinetic_energy = (price_delta ** 2) * np.log1p(current_vol)
                    sign_flip = np.sign(col_c[t]) != np.sign(col_c[t-1])
                    
                    if t >= lookback:
                        # Scan trailing baseline thresholds
                        recent_energy = []
                        for i in range(t-lookback, t):
                            p_d = col_a[i] - col_a[i-1]
                            v_m = col_v[i] if col_v[i] > 0 else 1.0
                            recent_energy.append((p_d ** 2) * np.log1p(v_m))
                        
                        energy_threshold = np.mean(recent_energy) * 1.3
                        
                        # Evaluating Sign Change Mystery Core
                        if sign_flip:
                            if kinetic_energy > energy_threshold:
                                current_regime = "10% Mode"
                            else:
                                current_regime = "90% Mode"
                        
                        col_d_state.append(current_regime)
                    else:
                        col_d_state.append("Warmup")
                
                # Map array outputs into clean storage state
                st.session_state.agca_robust_db = pd.DataFrame({
                    'Date_Time': [pd.to_datetime(x).strftime('%d %b %Y %H:%M') for x in frozen_data[time_key].values],
                    'Column A (Nifty Close)': [float(x) for x in col_a],
                    'Column B (Anchor)': [float(x) for x in col_b],
                    'Column C (Delta Variance)': [float(x) for x in col_c],
                    'Column D (Regime State)': col_d_state,
                    'NiftyBees Volume Mass': [float(x) for x in col_v]
                })
                
                st.sidebar.success("Database compiled seamlessly.")
                
        except Exception as ex:
            st.sidebar.error(f"Execution Refused: {str(ex)}")

# ==============================================================================
# 3. PRESENTATION GRID LAYER
# ==============================================================================
output_matrix = st.session_state.agca_robust_db.copy()

if not output_matrix.empty:
    st.write(f"### 📊 Engine Output Node: **{len(output_matrix)} Synchronized Data Streams Locked**")
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
        'NiftyBees Volume Mass': '{:,.0f}'
    }).map(style_column_d, subset=['Column D (Regime State)'])
    
    st.dataframe(styled_df, use_container_width=True)
else:
    st.info("Storage state empty. Press 'Force Core 1H Handshake' to pull 1-Hour candles.")
