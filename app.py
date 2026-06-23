import streamlit as st
import pandas as pd
import numpy as np
import datetime
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. BASE MATRIX CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Nifty Column G Live Handshake", layout="wide")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0b0f19 !important;
            color: #ffffff !important;
        }
        h1, h3, p, span, label { color: #ffffff !important; }
        .title-block {
            background: linear-gradient(90deg, #064e3b, #111827);
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #10b981;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-block">
        <h1>🎯 Nifty 50 Strict Cascade (Live Handshake & Freeze Engine)</h1>
        <p><b>System Logic:</b> Ek haath se completed data feed hoga, use freeze kiya jayega, aur dusre haath se Column G ka absolute signal compute hoga.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE MEMORY STORAGE (EXCEL MIRROR CORES)
# ==============================================================================
if 'matrix_storage' not in st.session_state:
    # Starting memory matrix benchmark placeholder
    st.session_state.matrix_storage = pd.DataFrame(columns=[
        'Date_Time', 'Column A', 'Column B', 'Column C', 'Column D', 'Column E', 'Column F', 'Column G'
    ])

# ==============================================================================
# 3. LIVE HANDSHAKE DATA INJECTION CONTROLS
# ==============================================================================
st.subheader("📥 Handshake Layer: Feed Next Completed Candle")

col_input_1, col_input_2, col_input_3 = st.columns([2, 2, 2])

with col_input_1:
    current_time_str = datetime.datetime.now().strftime('%d %b %Y %H:%M')
    input_time = st.text_input("⏰ Candle Close Timestamp", value=current_time_str)

with col_input_2:
    input_close = st.number_input("📈 Finalized Close Price (Column A)", min_value=0.0, value=23500.0, step=0.05, format="%.2f")

with col_input_3:
    st.markdown("<br>", unsafe_allow_html=True)
    submit_candle = st.button("⚡ Lock Entry & Generate Signal", use_container_width=True)

# System Reset to clean historical memory logs
if st.sidebar.button("🗑️ Reset Entire Local Storage Matrix"):
    st.session_state.matrix_storage = pd.DataFrame(columns=[
        'Date_Time', 'Column A', 'Column B', 'Column C', 'Column D', 'Column E', 'Column F', 'Column G'
    ])
    st.sidebar.success("Storage cleared successfully.")
    st.rerun()

# ==============================================================================
# 4. RECURSIVE TAIL MATHEMATICS PIPELINE (STABLE FREEZE)
# ==============================================================================
if submit_candle:
    history_df = st.session_state.matrix_storage.copy()
    mul = 0.0001
    
    A_new = float(input_close)
    
    # If it is the first row ever in memory, apply the zero gap initialization rule
    if len(history_df) == 0:
        B_new = A_new
        C_new = A_new - B_new
        D_new = C_new
        E_new = D_new
        F_new = E_new
        G_new = 0.0
    else:
        # Pull the exact last frozen index values directly from storage
        last_row = history_df.iloc[-1]
        
        B_prev = float(last_row['Column B'])
        C_prev = float(last_row['Column C'])
        D_prev = float(last_row['Column D'])
        E_prev = float(last_row['Column E'])
        F_prev = float(last_row['Column F'])
        
        # Calculate new nodes using locked historical state values
        B_new = B_prev + (mul * (A_new - B_prev))
        C_new = A_new - B_new
        D_new = D_prev + (mul * (C_new - D_prev))
        E_new = E_prev + (mul * (D_new - E_prev))
        F_new = F_prev + (mul * (E_new - F_prev))
        G_new = F_new - F_prev  # Direct row delta speed tracker
        
    # Packaging the brand new finalized record array
    new_record = {
        'Date_Time': input_time,
        'Column A': A_new,
        'Column B': B_new,
        'Column C': C_new,
        'Column D': D_new,
        'Column E': E_new,
        'Column F': F_new,
        'Column G': G_new
    }
    
    # Commit the row onto memory dataframes permanently
    st.session_state.matrix_storage = pd.concat([history_df, pd.DataFrame([new_record])], ignore_index=True)
    st.rerun()

# ==============================================================================
# 5. RENDER SYSTEM DISPLAY (LATEST COMMITTED HANDSHAKE ON TOP)
# ==============================================================================
display_df = st.session_state.matrix_storage.copy()

if not display_df.empty:
    st.subheader(f"📊 Active Frozen Database Matrix ({len(display_df)} Hard-Locked Rows)")
    
    # Reverse rows sequencing order for standard active grid view (Latest signal on top)
    inverted_view = display_df.iloc[::-1].reset_index(drop=True)
    
    st.dataframe(
        inverted_view.style.format({
            'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
            'Column D': '{:.4f}', 'Column E': '{:.4f}', 'Column F': '{:.4f}', 'Column G': '{:.6f}'
        }),
        use_container_width=True
    )
else:
    st.info("💡 Data storage matrix is empty. Nayi completed hour candle values upar enter karke system handshake load kijiye.")
