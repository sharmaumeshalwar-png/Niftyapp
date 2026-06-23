import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. BASE MATRIX CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Nifty Deep Automation Engine", layout="wide")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0b0f19 !important;
            color: #ffffff !important;
        }
        h1, h3, p, span, label { color: #ffffff !important; }
        .title-block {
            background: linear-gradient(90deg, #022c22, #0f172a);
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #06b6d4;
            margin-bottom: 20px;
        }
        .stButton>button {
            width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-block">
        <h1>🎯 Nifty 50 Strict Cascade (Deep Historical Auto-Handshake Engine)</h1>
        <p><b>Math Starting Anchor:</b> 01 June 2025 (Full History Load) | <b>Data Freeze Protocol:</b> Active<br>
        First time loading updates everything from June 2025. Subsequent ticks just handshake new candles automatically.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. PERMANENT DATA MEMORY INITIALIZATION
# ==============================================================================
if 'automated_matrix' not in st.session_state:
    st.session_state.automated_matrix = pd.DataFrame(columns=[
        'Date_Time', 'Column A', 'Column B', 'Column C', 'Column D', 'Column E', 'Column F', 'Column G'
    ])

# Control System Bars
st.sidebar.subheader("⚙️ System Control Matrix")
auto_check = st.sidebar.button("🔄 Handshake: Fetch Latest Candle")
reset_matrix = st.sidebar.button("🗑️ Reset Engine & Clear Data")

if reset_matrix:
    st.session_state.automated_matrix = pd.DataFrame(columns=[
        'Date_Time', 'Column A', 'Column B', 'Column C', 'Column D', 'Column E', 'Column F', 'Column G'
    ])
    st.sidebar.success("Database reset complete.")
    st.rerun()

# ==============================================================================
# 3. AUTOMATED DEEP PIPELINE (ONE-HAND DISPATCH / SECOND-HAND FREEZE)
# ==============================================================================
# Run deep sync if data is completely empty, or if user explicitly forces a handshake click
if len(st.session_state.automated_matrix) == 0 or auto_check:
    with st.spinner("Executing Deep Stream Analytics..."):
        try:
            # If storage is empty, load complete historic pipeline from June 2025
            if len(st.session_state.automated_matrix) == 0:
                raw_data = yf.download(tickers="^NSEI", start="2025-06-01", interval="1h", progress=False)
            else:
                # If data exists, fetch just the last 7 days to catch any new completed hours
                raw_data = yf.download(tickers="^NSEI", period="7d", interval="1h", progress=False)
                
            if not raw_data.empty:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    raw_data.columns = [col[0] for col in raw_data.columns]
                raw_data.columns = [str(col).strip().title() for col in raw_data.columns]
                
                raw_data = raw_data.dropna(subset=['Close']).sort_index(ascending=True)
                
                # Drop current unclosed dynamic hour candle to ensure absolute stability
                if len(raw_data) > 1:
                    completed_candles = raw_data.iloc[:-1].copy()
                else:
                    completed_candles = raw_data.copy()
                    
                completed_candles = completed_candles.reset_index()
                time_col = 'Datetime' if 'Datetime' in completed_candles.columns else completed_candles.columns[0]
                
                # Processing rows sequential injection
                for idx, row in completed_candles.iterrows():
                    row_time = pd.to_datetime(row[time_col]).strftime('%d %b %Y %H:%M')
                    row_close = float(row['Close'])
                    
                    history_df = st.session_state.automated_matrix.copy()
                    
                    # Deduplication Layer: Process calculation ONLY if timestamp does not exist in frozen memory
                    if row_time not in history_df['Date_Time'].values:
                        mul = 0.0001
                        
                        if len(history_df) == 0:
                            # Strict first row baseline lock rule (01 June 2025)
                            B_new = row_close
                            C_new = 0.0
                            D_new = 0.0
                            E_new = 0.0
                            F_new = 0.0
                            G_new = 0.0
                        else:
                            last_frozen = history_df.iloc[-1]
                            
                            B_prev = float(last_frozen['Column B'])
                            C_prev = float(last_frozen['Column C'])
                            D_prev = float(last_frozen['Column D'])
                            E_prev = float(last_frozen['Column E'])
                            F_prev = float(last_frozen['Column F'])
                            
                            # Sequential Cascade Formulations
                            B_new = B_prev + (mul * (row_close - B_prev))
                            C_new = row_close - B_new
                            D_new = D_prev + (mul * (C_new - D_prev))
                            E_new = E_prev + (mul * (D_new - E_prev))
                            F_new = F_prev + (mul * (E_new - F_prev))
                            G_new = F_new - F_prev
                            
                        # Format row bundle packet
                        appended_row = {
                            'Date_Time': row_time, 'Column A': row_close, 'Column B': B_new,
                            'Column C': C_new, 'Column D': D_new, 'Column E': E_new,
                            'Column F': F_new, 'Column G': G_new
                        }
                        
                        st.session_state.automated_matrix = pd.concat([st.session_state.automated_matrix, pd.DataFrame([appended_row])], ignore_index=True)
                        
        except Exception as e:
            st.error(f"Handshake interface dropped: {str(e)}")

# ==============================================================================
# 4. RENDER PRESENTATION MATRIX (LATEST REALTIME DATA ON TOP)
# ==============================================================================
final_df = st.session_state.automated_matrix.copy()

if not final_df.empty:
    st.subheader(f"📊 Active Frozen Database Matrix ({len(final_df)} Candle Signals Logged)")
    
    # Invert matrix grid display layer so recent logs appear first
    inverted_grid = final_df.iloc[::-1].reset_index(drop=True)
    
    st.dataframe(
        inverted_grid.style.format({
            'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
            'Column D': '{:.4f}', 'Column E': '{:.4f}', 'Column F': '{:.4f}', 'Column G': '{:.6f}'
        }),
        use_container_width=True
    )
else:
    st.warning("Core storage frame empty. Re-initiating June 2025 core matrix stream loop...")
