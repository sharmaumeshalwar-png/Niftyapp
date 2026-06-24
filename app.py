import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SCIENTIFIC UI THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AGCA Correlation Engine", layout="wide")

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
        <h1>🌌 AGCA Dual-Kalan Correlation Engine (1-Hour Resolution)</h1>
        <p><b>Column D:</b> NiftyBees Raw Vol | <b>Column E:</b> Volume Kalan Delta | <b>Column F:</b> Rolling Correlation between Price & Volume Deltas</p>
    </div>
""", unsafe_allow_html=True)

if 'agca_corr_db' not in st.session_state:
    st.session_state.agca_corr_db = pd.DataFrame()

st.sidebar.subheader("🔬 Control Room")
run_sync = st.sidebar.button("🔄 Execute Aligned Handshake")

# ==============================================================================
# 2. DUAL-KALAN & CORRELATION COMPUTATION MATRIX
# ==============================================================================
if len(st.session_state.agca_corr_db) == 0 or run_sync:
    with st.spinner("Locking arrays and calculating cross-correlation..."):
        try:
            nifty_raw = yf.download(tickers="^NSEI", period="2mo", interval="1h", progress=False)
            bees_raw = yf.download(tickers="NIFTYBEES.NS", period="2mo", interval="1h", progress=False)
            
            if not nifty_raw.empty and not bees_raw.empty:
                if isinstance(nifty_raw.columns, pd.MultiIndex):
                    nifty_raw.columns = [col[0] for col in nifty_raw.columns]
                if isinstance(bees_raw.columns, pd.MultiIndex):
                    bees_raw.columns = [col[0] for col in bees_raw.columns]
                
                nifty_raw.columns = [str(col).strip().title() for col in nifty_raw.columns]
                bees_raw.columns = [str(col).strip().title() for col in bees_raw.columns]
                
                nifty_raw.index = pd.to_datetime(nifty_raw.index)
                bees_raw.index = pd.to_datetime(bees_raw.index)
                
                bees_vol_series = bees_raw['Volume'].dropna()
                nifty_raw['Real_Vol'] = nifty_raw.index.map(
                    lambda x: float(bees_vol_series.loc[x]) if x in bees_vol_series.index else float(bees_vol_series.median())
                )
                
                nifty_df = nifty_raw[['Close', 'Real_Vol']].dropna().sort_index()
                frozen_data = nifty_df.iloc[:-1].copy() if len(nifty_df) > 1 else nifty_df.copy()
                frozen_data = frozen_data.reset_index()
                
                time_key = 'Datetime' if 'Datetime' in frozen_data.columns else frozen_data.columns[0]
                total_elements = len(frozen_data)
                
                # Arrays Extraction
                col_a = np.array(frozen_data['Close'].values, dtype=float).flatten()
                col_d = np.array(frozen_data['Real_Vol'].values, dtype=float).flatten() # Column D = Raw Volume
                
                # Allocation for Price Kalan Formula
                col_b = np.zeros(total_elements, dtype=float)
                col_c = np.zeros(total_elements, dtype=float)
                
                # Allocation for Volume Kalan Formula
                vol_anchor = np.zeros(total_elements, dtype=float)
                col_e = np.zeros(total_elements, dtype=float) # Column E = Volume Delta
                
                # Base Seeds
                if total_elements > 0:
                    col_b[0] = float(col_a[0])
                    col_c[0] = 0.0
                    
                    vol_anchor[0] = float(col_d[0])
                    col_e[0] = 0.0
                
                multiplier = 0.0001
                
                # Dual Kalan Processing Loop
                for t in range(1, total_elements):
                    # 1. Price Kalan Formula (A -> B -> C)
                    col_b[t] = col_b[t-1] + (multiplier * (col_a[t] - col_b[t-1]))
                    col_c[t] = col_a[t] - col_b[t].round(4)
                    
                    # 2. Volume Kalan Formula (D -> Vol_Anchor -> E)
                    vol_anchor[t] = vol_anchor[t-1] + (multiplier * (col_d[t] - vol_anchor[t-1]))
                    col_e[t] = col_d[t] - vol_anchor[t]
                
                # Step 3: Rolling Correlation Matrix Generation (Column F)
                temp_df = pd.DataFrame({'C': col_c, 'E': col_e})
                col_f = temp_df['C'].rolling(window=14, min_periods=1).corr(temp_df['E']).fillna(0.0).values
                
                # Store structural outputs cleanly
                st.session_state.agca_corr_db = pd.DataFrame({
                    'Date_Time': [pd.to_datetime(x).strftime('%d %b %Y %H:%M') for x in frozen_data[time_key].values],
                    'Column A (Nifty Close)': col_a.round(2),
                    'Column B (Price Anchor)': col_b.round(4),
                    'Column C (Price Delta)': col_c.round(4),
                    'Column D (NiftyBees Vol)': col_d.astype(int),
                    'Column E (Volume Delta)': col_e.round(4),
                    'Column F (Correlation)': col_f.round(4)
                })
                st.sidebar.success("Dual Kalan Matrix Synchronized!")
        except Exception as ex:
            st.error(f"Core breakdown: {str(ex)}")

# ==============================================================================
# 3. PRESENTATION GRID LAYER
# ==============================================================================
output_matrix = st.session_state.agca_corr_db.copy()

if not output_matrix.empty:
    st.write(f"### 📊 Processed Core Stream: **{len(output_matrix)} Data Intervals Decoded**")
    inverted_view = output_matrix.iloc[::-1].reset_index(drop=True)
    
    def style_correlation(val):
        if val > 0.5:
            return 'background-color: #14532d; color: #a7f3d0; font-weight: bold;' # Strong positive sync
        elif val < -0.5:
            return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;' # Strong negative divergence
        return ''

    try:
        styled_df = inverted_view.style.map(style_correlation, subset=['Column F (Correlation)'])
    except AttributeError:
        styled_df = inverted_view.style.applymap(style_correlation, subset=['Column F (Correlation)'])
        
    st.dataframe(styled_df, use_container_width=True)
else:
    st.info("Click 'Execute Aligned Handshake' to initialize dual oscillator tracking.")
