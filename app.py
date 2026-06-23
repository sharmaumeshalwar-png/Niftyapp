import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. BASE MATRIX CONFIGURATION (MOBILE FRIENDLY VIEW)
# ==============================================================================
st.set_page_config(page_title="Nifty Live Automation", layout="wide")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0b0f19 !important;
            color: #ffffff !important;
        }
        h1, h3, p, span, label { color: #ffffff !important; }
        .title-block {
            background: linear-gradient(90deg, #0f172a, #1e1b4b);
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #3b82f6;
            margin-bottom: 20px;
        }
        /* Mobile adjustment for buttons */
        .stButton>button {
            width: 100% !important;
            white-space: normal !important;
            word-wrap: break-word !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-block">
        <h1>🚀 Nifty 50 Strict Cascade (Fully Automated Handshake)</h1>
        <p><b>Logic:</b> Script automated command se yfinance se bar-bar check karegi. Candle complete hote hi data freeze hoga aur Column G execute hoga. Back-data change nahi hoga!</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. PERMANENT STATE MEMORY CORES
# ==============================================================================
if 'automated_matrix' not in st.session_state:
    st.session_state.automated_matrix = pd.DataFrame(columns=[
        'Date_Time', 'Column A', 'Column B', 'Column C', 'Column D', 'Column E', 'Column F', 'Column G'
    ])

# Sidebar Controls for User Trigger
st.sidebar.subheader("⚙️ Control Dashboard")
auto_check = st.sidebar.button("🔄 Check & Fetch Latest Completed Candle")
clear_data = st.sidebar.button("🗑️ Clear Storage Reset Matrix")

if clear_data:
    st.session_state.automated_matrix = pd.DataFrame(columns=[
        'Date_Time', 'Column A', 'Column B', 'Column C', 'Column D', 'Column E', 'Column F', 'Column G'
    ])
    st.sidebar.success("Database wiped clean!")
    st.rerun()

# ==============================================================================
# 3. BACKGROUND FETCH & HARD RECURSIVE LOCK (ONE-HAND HANDSHAKE)
# ==============================================================================
if auto_check or len(st.session_state.automated_matrix) == 0:
    with st.spinner("Checking market terminal..."):
        try:
            # Safe interval window loading
            raw_data = yf.download(tickers="^NSEI", period="5d", interval="1h", progress=False)
            
            if not raw_data.empty:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    raw_data.columns = [col[0] for col in raw_data.columns]
                raw_data.columns = [str(col).strip().title() for col in raw_data.columns]
                
                raw_data = raw_data.dropna(subset=['Close']).sort_index(ascending=True)
                
                # [CRITICAL LOGIC] Current running unclosed hourly candle dropped to avoid re-calculation
                if len(raw_data) > 1:
                    completed_candles = raw_data.iloc[:-1].copy()
                else:
                    completed_candles = raw_data.copy()
                
                completed_candles = completed_candles.reset_index()
                time_col = 'Datetime' if 'Datetime' in completed_candles.columns else completed_candles.columns[0]
                
                # Process latest row fetched from exchange
                for idx, row in completed_candles.iterrows():
                    row_time = pd.to_datetime(row[time_col]).strftime('%d %b %Y %H:%M')
                    row_close = float(row['Close'])
                    
                    history_df = st.session_state.automated_matrix.copy()
                    
                    # Check if this hour record time is already hard-locked in system memory
                    if row_time not in history_df['Date_Time'].values:
                        mul = 0.0001
                        
                        if len(history_df) == 0:
                            # Base seed initialization rule
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
                            
                            # Sequential Cascade Math Loop
                            B_new = B_prev + (mul * (row_close - B_prev))
                            C_new = row_close - B_new
                            D_new = D_prev + (mul * (C_new - D_prev))
                            E_new = E_prev + (mul * (D_new - E_prev))
                            F_new = F_prev + (mul * (E_new - F_prev))
                            G_new = F_new - F_prev
                            
                        # Package stable dataset array row
                        appended_row = {
                            'Date_Time': row_time, 'Column A': row_close, 'Column B': B_new,
                            'Column C': C_new, 'Column D': D_new, 'Column E': E_new,
                            'Column F': F_new, 'Column G': G_new
                        }
                        
                        st.session_state.automated_matrix = pd.concat([st.session_state.automated_matrix, pd.DataFrame([appended_row])], ignore_index=True)
            
        except Exception as e:
            st.error(f"Handshake connection error: {str(e)}")

# ==============================================================================
# 4. DATA PRESENTATION LAYER (LATEST SIGNALS ALWAYS SIT ON TOP)
# ==============================================================================
final_df = st.session_state.automated_matrix.copy()

if not final_df.empty:
    st.subheader(f"📊 Auto-Locked Dataset Rows Matrix ({len(final_df)} Candle Signals)")
    
    # Invert rows array direction for responsive viewing (Latest timestamp on top)
    inverted_grid = final_df.iloc[::-1].reset_index(drop=True)
    
    st.dataframe(
        inverted_grid.style.format({
            'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
            'Column D': '{:.4f}', 'Column E': '{:.4f}', 'Column F': '{:.4f}', 'Column G': '{:.6f}'
        }),
        use_container_width=True
    )
else:
    st.info("💡 Automation pool active. Market closing updates fetch complete waiting loop trigger.")
