import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty Micro-Hull Structural System", layout="wide")
st.title("🎯 Nifty 50 Dual-Stage Micro-Structural Velocity Engine")
st.write("De-trending Nifty 50 (^NSEI) from May 1, 2026. Catching 90% Traps via Pure Iterative Memory Gaps.")

# Fetch 1-Hour Accurate Nifty 50 Data safely
@st.cache_data(ttl=300)
def load_pure_data():
    df_raw = yf.download(tickers="^NSEI", start="2026-05-01", interval="1h")
    if df_raw.empty:
        return pd.DataFrame()
        
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
        
    # Drop empty or corrupted data rows instantly
    df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    return df_raw

df = load_pure_data()

if not df.empty:
    df = df.reset_index()
    
    # 1. Column D: Date and Time Formatter
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    # 2. Column A: Exact Formula -> (High + Low) / 2
    df['Column A'] = (df['High'] + df['Low']) / 2
    
    # 3. Column B: Your Original Base Loop (Multiplier = 0.0001)
    multiplier_base = 0.0001
    n_col_b = np.zeros(len(df))
    
    if len(df) > 0:
        n_col_b[0] = df['Column A'].iloc[0]
        
    for i in range(1, len(df)):
        n_col_b[i] = n_col_b[i-1] + (multiplier_base * (df['Column A'].iloc[i] - n_col_b[i-1]))
        
    df['Column B'] = n_col_b
    
    # 4. Column C: Exact Formula -> A - B Deviation
    df['Column C'] = df['Column A'] - df['Column B']
    
    # ---------------------------------------------------------
    # 🛠️ PURE MATHEMATICAL MICRO-LEVEL TRACKING SEQUENCES
    # ---------------------------------------------------------
    # Sequence B1 (Fast Track Multiplier = 0.01)
    m_fast = 0.01
    b1_seq = np.zeros(len(df))
    
    # Sequence B2 (Slow Track Multiplier = 0.0001)
    m_slow = 0.0001
    b2_seq = np.zeros(len(df))
    
    if len(df) > 0:
        b1_seq[0] = df['Column A'].iloc[0]
        b2_seq[0] = df['Column A'].iloc[0]
        
    for i in range(1, len(df)):
        b1_seq[i] = b1_seq[i-1] + (m_fast * (df['Column A'].iloc[i] - b1_seq[i-1]))
        b2_seq[i] = b2_seq[i-1] + (m_slow * (df['Column A'].iloc[i] - b2_seq[i-1]))
        
    df['B1_Fast'] = b1_seq
    df['B2_Slow'] = b2_seq
    
    # Calculate the Structural Micro Gap
    df['Micro_Gap'] = df['B1_Fast'] - df['B2_Slow']
    
    # Sequence B3 (Catalyst Tracker on the Micro Gap itself - Multiplier = 0.05)
    m_catalyst = 0.05
    b3_seq = np.zeros(len(df))
    
    if len(df) > 0:
        b3_seq[0] = df['Micro_Gap'].iloc[0]
        
    for i in range(1, len(df)):
        b3_seq[i] = b3_seq[i-1] + (m_catalyst * (df['Micro_Gap'].iloc[i] - b3_seq[i-1]))
        
    df['B3_Catalyst'] = b3_seq
    
    # ---------------------------------------------------------
    # 🎯 COLUMN E: DUAL-STAGE VELOCITY TRAP GRID INTERSECTION
    # ---------------------------------------------------------
    status_list = ["Baseline System Boot"]
    
    for i in range(1, len(df)):
        curr_c = df['Column C'].iloc[i]
        gap = df['Micro_Gap'].iloc[i]
        catalyst = df['B3_Catalyst'].iloc[i]
        
        # Micro structural validation cross check
        micro_bullish = gap > catalyst
        
        # --- THE REFINED 90% VS 10% PRICE ACTION MATRIX ---
        if curr_c > 0:  # Column C is Plus (+)
            if micro_bullish:  # Flow is structurally accelerating upward -> 10% Real Move
                status_list.append("🟢 TRUE BULLISH MOMENTUM (Micro-Engine Verified)")
            else:  # Flow is collapsing inside the sequence -> THE 90% CALL TRAP DETECTED!
                status_list.append("⚠️ 90% FAKE UPMOVE TRAP (Micro Decay!)")
                
        else:  # Column C is Minus (-)
            if not micro_bullish:  # Flow is structurally collapsing downward -> Real Breakdown
                status_list.append("🔴 TRUE BEARISH CRASH (Micro-Engine Verified)")
            else:  # Flow is accumulating traction below -> THE 90% PUT TRAP DETECTED!
                status_list.append("⚠️ 90% FAKE DOWNMOVE TRAP (Micro Accumulation!)")
                
    df['Column E'] = status_list
    
    # Trim output display boundary cleanly for 1st May 2026 onwards
    df = df[df['Raw_Date'] >= '2026-05-01'].copy()
    
    # Reverse dataset array to throw latest 1-Hour rows at the top of the grid
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'Micro_Gap', 'B3_Catalyst', 'Column E']].copy()
    show_df = show_df.iloc[::-1]
    
    # Distinct Dashboard Layout Hex-Theming
    def color_trap_grid(val):
        if "🟢" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "🔴" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "⚠️" in val: return 'background-color: #d35400; color: white; font-weight: bold;'
        if "💤" in val: return 'background-color: #34495e; color: #bdc3c7;'
        return ''

    # Precision Decimal Formatter Block
    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
        'Micro_Gap': '{:.4f}', 'B3_Catalyst': '{:.4f}'
    }).map(color_trap_grid, subset=['Column E']), use_container_width=True)
else:
    st.error("Data ingestion failure: Data stream structure couldn't be mounted on Streamlit.")
