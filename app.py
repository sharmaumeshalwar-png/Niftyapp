import streamlit as st
import pandas as pd
import numpy as np
import time

# Page configuration
st.set_page_config(page_title="Nifty Live Custom Tracker", layout="wide")
st.title("📊 Nifty 1-Minute Live Custom Tracker")
st.write("Real-time formula evaluation & Option Selling / Blast alerts.")

# Initialize sessions for database storage
if 'df_data' not in st.session_state:
    # Starting baseline row to avoid index errors for previous rows
    st.session_state.df_data = pd.DataFrame([{
        "Row": 1,
        "Col_A_Close": 23500.0,
        "High": 23510.0,
        "Low": 23490.0,
        "Col_B_Avg": 23500.0,      # (High + Low) / 2
        "Col_M": 10.0,              # Placeholders for your custom M & N columns
        "Col_N": 5.0,               
        "Col_C_Formula": 23500.0,  
        "Col_D_Diff": 0.0,
        "Signal": "WAIT"
    }])

# Sidebar controls
st.sidebar.header("System Controls")
live_toggle = st.sidebar.toggle("Start Live Feed", value=True)
refresh_rate = st.sidebar.slider("Refresh Rate (Seconds)", 1, 10, 2)

placeholder = st.empty()

# 8-Step Loop Execution
while live_toggle:
    # 1. Fetch current dataframe
    df = st.session_state.df_data.copy()
    prev_row = df.iloc[-1]
    next_row_num = len(df) + 1
    
    # 2. Simulate Live Nifty 1-Min Candle Data (Step 1 & Step 2)
    base_price = prev_row["Col_A_Close"] + np.random.uniform(-15, 15)
    high_val = base_price + np.random.uniform(2, 10)
    low_val = base_price - np.random.uniform(2, 10)
    close_val = np.random.uniform(low_val, high_val)
    
    # 3. Step 3: Column B = (High + Low) / 2
    col_b_avg = (high_val + low_val) / 2
    
    # Simulate custom metrics for M and N to avoid blank data
    col_m_val = np.random.uniform(-5, 5)
    col_n_val = np.random.uniform(-5, 5)
    
    # 4. Step 4: Column C = B_current + 0.001 * (M_current - N_previous)
    # Excel terminology translated: B2_current + 0.001 * (M3 - N2)
    col_c_formula = col_b_avg + 0.001 * (col_m_val - prev_row["Col_N"])
    
    # 5. Step 5: Column D = A (Close) - B (Avg)
    col_d_diff = close_val - col_b_avg
    
    # 6. Step 6 & 7: 90% Reversal vs 10% Blast Detection Logic
    # Testing direction change from the previous row's difference
    if np.sign(col_d_diff) != np.sign(prev_row["Col_D_Diff"]):
        # If velocity spikes abnormally, it's a 10% Blast Move
        if abs(col_d_diff) > 8.0:
            signal = "💥 BLAST MOVE"
        else:
            signal = "🎯 OPTION SELL"
    else:
        signal = "WAIT"
        
    # Create new row dictionary
    new_row = {
        "Row": next_row_num,
        "Col_A_Close": round(close_val, 2),
        "High": round(high_val, 2),
        "Low": round(low_val, 2),
        "Col_B_Avg": round(col_b_avg, 2),
        "Col_M": round(col_m_val, 4),
        "Col_N": round(col_n_val, 4),
        "Col_C_Formula": round(col_c_formula, 4),
        "Col_D_Diff": round(col_d_diff, 2),
        "Signal": signal
    }
    
    # Append and keep tracking
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    st.session_state.df_data = df
    
    # 8. Step 8: Render Clean Layout Output
    with placeholder.container():
        # Metric alerts on top
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Current Nifty", f"₹ {new_row['Col_A_Close']}")
        kpi2.metric("Latest Signal", new_row['Signal'])
        kpi3.metric("Current Gap (Col D)", f"{new_row['Col_D_Diff']}")
        
        # Style dataframe for scannability
        def style_signals(val):
            if val == "🎯 OPTION SELL":
                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif val == "💥 BLAST MOVE":
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
            return ''

        styled_df = df.tail(20).style.applymap(style_signals, subset=['Signal'])
        
        st.subheader("Live Tracker Sheet (Last 20 Rows)")
        st.dataframe(styled_df, use_container_width=True, height=500)
        
    time.sleep(refresh_rate)

if not live_toggle:
    st.info("Live feed stopped. Toggle 'Start Live Feed' to resume tracking.")
