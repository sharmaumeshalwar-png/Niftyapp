import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="WMA Simulation Test", layout="wide")
st.title("🧪 Artificial Price 12-WMA Simulation Engine")
st.write("Target: Check how linear weights prevent fake signals using sample inputs.")

# 1. Generating Artificial Price Stream (Starting from ₹100)
# Simulating a pump up to 250, then a sudden consolidation/whipsaw to test trap zone
sample_highs = [100, 120, 130, 140, 150, 160, 165, 170, 175, 180, 190, 200, 210, 220, 230, 240, 250, 248, 247, 249, 246, 252]
sample_lows  = [90,  110, 115, 125, 135, 145, 150, 155, 160, 165, 175, 185, 195, 205, 215, 225, 235, 230, 228, 232, 229, 236]
sample_closes = [95,  115, 122, 132, 142, 152, 158, 162, 168, 172, 182, 192, 202, 212, 222, 232, 242, 240, 238, 241, 235, 250]

sim_df = pd.DataFrame({
    'High': sample_highs,
    'Low': sample_lows,
    'Close': sample_closes
})

# 2. Arithmetic WMA Logic Setup
wma_weights = np.arange(12, 0, -1) # [12, 11, 10, ... 1]
wma_sum = np.sum(wma_weights)       # 78

def calc_wma(series):
    return series.rolling(window=12).apply(lambda x: np.sum(x * wma_weights) / wma_sum, raw=True)

sim_df['WMA_High_Tunnel'] = calc_wma(sim_df['High'])
sim_df['WMA_Low_Tunnel'] = calc_wma(sim_df['Low'])

# 3. Signal Generation Tree
status_log = []
for idx in range(len(sim_df)):
    c_close = sim_df['Close'].iloc[idx]
    h_tunnel = sim_df['WMA_High_Tunnel'].iloc[idx]
    l_tunnel = sim_df['WMA_Low_Tunnel'].iloc[idx]
    
    if pd.isna(h_tunnel) or pd.isna(l_tunnel):
        status_log.append("🔄 LOADING MATRIX (Need 12 Candles)")
    elif c_close > h_tunnel:
        status_log.append("🟢 BUY SIGNAL (Breakout)")
    elif c_close < l_tunnel:
        status_log.append("🔴 SELL SIGNAL (Breakdown)")
    else:
        status_log.append("⏳ TRAP ZONE (Price Caught Inside)")

sim_df['System_Output'] = status_log

# Format Output for clear UI scannability
display_df = sim_df.copy()
display_df['WMA_High_Tunnel'] = display_df['WMA_High_Tunnel'].round(2)
display_df['WMA_Low_Tunnel'] = display_df['WMA_Low_Tunnel'].round(2)
display_df = display_df.iloc[::-1] # Reverse to keep latest on top

st.subheader("📋 Live Simulation Matrix Log")
st.dataframe(display_df, use_container_width=True)
