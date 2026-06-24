import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SCIENTIFIC UI THEME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AGCA Pure EMA Core", layout="wide")

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
        <h1>🌌 AGCA Pure Price & EMA Handshake (1-Hour Core)</h1>
        <p><b>Variables Preserved:</b> Column A (Close), Column B (Anchor), Column C (Delta)<br>
        <b>Column D Filter Node:</b> Aligns 20/50 EMA filters precisely on Column C Sign Transitions.</p>
    </div>
""", unsafe_allow_html=True)

if 'agca_ema_pure_db' not in st.session_state:
    st.session_state.agca_ema_pure_db = pd.DataFrame()

st.sidebar.subheader("🔬 Core Controls")
run_sync = st.sidebar.button("🔄 Sync 1-Hour Price Matrix")

# ==============================================================================
# 2. SEAMLESS PRICE & EMA COMPUTATION MATRIX
# ==============================================================================
if len(st.session_state.agca_ema_pure_db) == 0 or run_sync:
    with st.spinner("Locking price arrays and rendering EMA layers..."):
        try:
            nifty_raw = yf.download(tickers="^NSEI", period="2mo", interval="1h", progress=False)
            
            if not nifty_raw.empty:
                if isinstance(nifty_raw.columns, pd.MultiIndex):
                    nifty_raw.columns = [col[0] for col in nifty_raw.columns]
                
                nifty_raw.columns = [str(col).strip().title() for col in nifty_raw.columns]
                nifty_raw.index = pd.to_datetime(nifty_raw.index)
                
                nifty_df = nifty_raw[['Close']].dropna().sort_index()
                
                # Dynamic technical indicators math injection
                nifty_df['EMA_20'] = nifty_df['Close'].ewm(span=20, adjust=False).mean()
                nifty_df['EMA_50'] = nifty_df['Close'].ewm(span=50, adjust=False).mean()
                
                # Freeze current active live candle
                frozen_data = nifty_df.iloc[:-1].copy() if len(nifty_df) > 1 else nifty_df.copy()
                frozen_data = frozen_data.reset_index()
                
                time_key = 'Datetime' if 'Datetime' in frozen_data.columns else frozen_data.columns[0]
                total_elements = len(frozen_data)
                
                # Arrays Allocation
                col_a = np.array(frozen_data['Close'].values, dtype=float).flatten()
                ema20_arr = np.array(frozen_data['EMA_20'].values, dtype=float).flatten()
                ema50_arr = np.array(frozen_data['EMA_50'].values, dtype=float).flatten()
                
                col_b = np.zeros(total_elements, dtype=float)
                col_c = np.zeros(total_elements, dtype=float)
                
                col_d_state = []
                
                if total_elements > 0:
                    col_b[0] = float(col_a[0])
                    col_c[0] = 0.0
                    col_d_state.append("Warmup")
                
                multiplier = 0.0001
                
                # Process Matrix Loop
                for t in range(1, total_elements):
                    # Kalan Pricing Math Rules (Step 1 to Step 8 Tracking)
                    col_b[t] = col_b[t-1] + (multiplier * (col_a[t] - col_b[t-1]))
                    col_c[t] = col_a[t] - col_b[t]
                    
                    # Sign Change Scanner Mechanism
                    current_sign = np.sign(col_c[t])
                    prev_sign = np.sign(col_c[t-1])
                    
                    current_close = col_a[t]
                    c_ema20 = ema20_arr[t]
                    c_ema50 = ema50_arr[t]
                    
                    # Rule Mapping Filters
                    if prev_sign == -1 and current_sign == 1:
                        # Sign changed from Minus to Plus
                        if current_close > c_ema20 and current_close > c_ema50:
                            col_d_state.append("BULLISH (Above 20/50 EMA)")
                        else:
                            col_d_state.append("No Structure")
                            
                    elif prev_sign == 1 and current_sign == -1:
                        # Sign changed from Plus to Minus
                        if current_close < c_ema20 and current_close < c_ema50:
                            col_d_state.append("BEARISH (Below 20/50 EMA)")
                        else:
                            col_d_state.append("No Structure")
                    else:
                        # No active transition sequence triggered on this candle
                        col_d_state.append("No Structure")
                
                # Clean Output Frame Synthesis
                st.session_state.agca_ema_pure_db = pd.DataFrame({
                    'Date_Time': [pd.to_datetime(x).strftime('%d %b %Y %H:%M') for x in frozen_data[time_key].values],
                    'Column A (Nifty Close)': col_a.round(2),
                    'Column B (Anchor)': col_b.round(4),
                    'Column C (Delta Variance)': col_c.round(4),
                    'Column D (EMA Filter State)': col_d_state
                })
                st.sidebar.success("EMA System Core Synchronized!")
        except Exception as ex:
            st.error(f"Core breakdown: {str(ex)}")

# ==============================================================================
# 3. PRESENTATION GRID LAYER
# ==============================================================================
output_matrix = st.session_state.agca_ema_pure_db.copy()

if not output_matrix.empty:
    st.write(f"### 📊 Processed Pricing Stream: **{len(output_matrix)} 1-Hour Snapshots Decoded**")
    inverted_view = output_matrix.iloc[::-1].reset_index(drop=True)
    
    def style_column_d(val):
        if "BULLISH" in str(val):
            return 'background-color: #14532d; color: #a7f3d0; font-weight: bold; border: 1px solid #10b981;'
        elif "BEARISH" in str(val):
            return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold; border: 1px solid #ef4444;'
        return ''

    try:
        styled_df = inverted_view.style.map(style_column_d, subset=['Column D (EMA Filter State)'])
    except AttributeError:
        styled_df = inverted_view.style.applymap(style_column_d, subset=['Column D (EMA Filter State)'])
        
    st.dataframe(styled_df, use_container_width=True)
else:
    st.info("Click 'Sync 1-Hour Price Matrix' to activate structural EMA scan.")
