import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SCIENTIFIC UI THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AGCA 1H Core Fixed", layout="wide")

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
            border: 1px solid #10b981;
            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="decoder-block">
        <h1>🌌 AGCA Pure 1-Hour Engine (Stable Release Core)</h1>
        <p><b>Timeframe Matrix:</b> 1-Hour Structural Candles | <b>Data Safe Frame:</b> Rolling 2-Month Windows Compliance<br>
        Column D Isolates Core States: <span style='color:#fca5a5; font-weight:bold;'>10% Mode (Red)</span> or <span style='color:#6ee7b7; font-weight:bold;'>90% Mode (Green)</span>.</p>
    </div>
""", unsafe_allow_html=True)

if 'agca_1h_fixed_db' not in st.session_state:
    st.session_state.agca_1h_fixed_db = pd.DataFrame()

st.sidebar.subheader("🔬 Control Room")
run_sync = st.sidebar.button("🔄 Sync Aligned 1-Hour Matrix")

# ==============================================================================
# 2. ROBUST DUAL-ASSET 1-HOUR COMPUTATION MATRIX
# ==============================================================================
if len(st.session_state.agca_1h_fixed_db) == 0 or run_sync:
    with st.spinner("Locking 1-Hour candles and fixing index arrays..."):
        try:
            nifty_raw = yf.download(tickers="^NSEI", period="2mo", interval="1h", progress=False)
            bees_raw = yf.download(tickers="NIFTYBEES.NS", period="2mo", interval="1h", progress=False)
            
            if not nifty_raw.empty and not bees_raw.empty:
                # Robust Multi-index Header Flattening Protection
                if isinstance(nifty_raw.columns, pd.MultiIndex):
                    nifty_raw.columns = [col[0] for col in nifty_raw.columns]
                if isinstance(bees_raw.columns, pd.MultiIndex):
                    bees_raw.columns = [col[0] for col in bees_raw.columns]
                
                nifty_raw.columns = [str(col).strip().title() for col in nifty_raw.columns]
                bees_raw.columns = [str(col).strip().title() for col in bees_raw.columns]
                
                # Aligning Timezones and Indexes
                nifty_raw.index = pd.to_datetime(nifty_raw.index)
                bees_raw.index = pd.to_datetime(bees_raw.index)
                
                # Strict 0% Row Drop Mapping Logic
                bees_vol_series = bees_raw['Volume'].dropna()
                nifty_raw['Real_Vol'] = nifty_raw.index.map(
                    lambda x: float(bees_vol_series.loc[x]) if x in bees_vol_series.index else float(bees_vol_series.median())
                )
                
                nifty_df = nifty_raw[['Close', 'Real_Vol']].dropna().sort_index()
                
                # Freeze current active running candle to avoid flash signals
                frozen_data = nifty_df.iloc[:-1].copy() if len(nifty_df) > 1 else nifty_df.copy()
                frozen_data = frozen_data.reset_index()
                
                time_key = 'Datetime' if 'Datetime' in frozen_data.columns else frozen_data.columns[0]
                total_elements = len(frozen_data)
                
                col_a = np.array(frozen_data['Close'].values, dtype=float).flatten()
                col_v = np.array(frozen_data['Real_Vol'].values, dtype=float).flatten()
                col_b = np.zeros(total_elements, dtype=float)
                col_c = np.zeros(total_elements, dtype=float)
                
                col_d_state = []
                
                # Baseline Seed Initialization
                if total_elements > 0:
                    col_b[0] = float(col_a[0])
                    col_c[0] = 0.0
                    col_d_state.append("Warmup")
                
                multiplier = 0.0001
                lookback = 14  
                current_regime = "90% Mode"
                
                # Production Loop Iterator
                for t in range(1, total_elements):
                    current_a = float(col_a[t])
                    prev_b = float(col_b[t-1])
                    
                    # Core Mathematical Matrix
                    col_b[t] = prev_b + (multiplier * (current_a - prev_b))
                    col_c[t] = current_a - col_b[t]
                    
                    price_delta = current_a - float(col_a[t-1])
                    bees_volume = float(col_v[t]) if float(col_v[t]) > 0 else 1.0
                    
                    # Kinetic Energy Formulation
                    kinetic_energy = (price_delta ** 2) * np.log1p(bees_volume)
                    sign_flip = np.sign(col_c[t]) != np.sign(col_c[t-1])
                    
                    if t >= lookback:
                        recent_energy = []
                        for i in range(t-lookback, t):
                            p_d = col_a[i] - col_a[i-1]
                            v_m = col_v[i] if col_v[i] > 0 else 1.0
                            recent_energy.append((p_d ** 2) * np.log1p(v_m))
                        
                        energy_threshold = np.mean(recent_energy) * 1.3
                        
                        # Evaluating Sign Change Core Mystery
                        if sign_flip:
                            if kinetic_energy > energy_threshold:
                                current_regime = "10% Mode"
                            else:
                                current_regime = "90% Mode"
                        
                        col_d_state.append(current_regime)
                    else:
                        col_d_state.append("Warmup")
                
                # Compilation to dataframe
                st.session_state.agca_1h_fixed_db = pd.DataFrame({
                    'Date_Time': [pd.to_datetime(x).strftime('%d %b %Y %H:%M') for x in frozen_data[time_key].values],
                    'Column A (Nifty Close)': [float(x) for x in col_a],
                    'Column B (Anchor)': [float(x) for x in col_b],
                    'Column C (Delta Variance)': [float(x) for x in col_c],
                    'Column D (Regime State)': col_d_state,
                    'NiftyBees Vol': [float(x) for x in col_v]
                })
                
                st.sidebar.success("Database compiled successfully!")
            else:
                st.error("API returned empty values. Refreshing required.")
                
        except Exception as ex:
            st.error(f"Core execution breakdown: {str(ex)}")

# ==============================================================================
# 3. PRESENTATION GRID LAYER (FIXED TO BYPASS LINE 157 FORMAT ERRORS)
# ==============================================================================
output_matrix = st.session_state.agca_1h_fixed_db.copy()

if not output_matrix.empty:
    st.write(f"### 📊 Active Stream Node: **{len(output_matrix)} Hourly Candles Processed**")
    inverted_view = output_matrix.iloc[::-1].reset_index(drop=True)
    
    # Precise float rounding directly onto the dataframe before passing to style engine
    inverted_view['Column A (Nifty Close)'] = inverted_view['Column A (Nifty Close)'].round(2)
    inverted_view['Column B (Anchor)'] = inverted_view['Column B (Anchor)'].round(4)
    inverted_view['Column C (Delta Variance)'] = inverted_view['Column C (Delta Variance)'].round(4)
    inverted_view['NiftyBees Vol'] = inverted_view['NiftyBees Vol'].astype(int)

    def style_column_d(val):
        if "10%" in str(val):
            return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold; border: 1px solid #ef4444;'
        elif "90%" in str(val):
            return 'background-color: #14532d; color: #a7f3d0;'
        return ''

    # Bulletproof Grid Renderer (Bypasses all method limitations of older/newer Pandas packages)
    try:
        styled_df = inverted_view.style.map(style_column_d, subset=['Column D (Regime State)'])
    except AttributeError:
        styled_df = inverted_view.style.applymap(style_column_d, subset=['Column D (Regime State)'])
    
    st.dataframe(styled_df, use_container_width=True)
else:
    st.info("System storage offline. Click 'Sync Aligned 1-Hour Matrix' control panel button.")
