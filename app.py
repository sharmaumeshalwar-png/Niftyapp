import streamlit as st
import numpy as np
import pandas as pd
import requests

# Page Configuration
st.set_page_config(page_title="BTC Breakout Core Engine", layout="wide")
st.title("⚡ BTC Live 1-Hour Standalone Breakout Engine")
st.write("🎯 **Pure Math Mode (Zero Library Error):** 50:50 Split + Multi-Node Redundancy + Dynamic Zone Tracker")

# =====================================================================
# MATHEMATICAL ENGINE (Flexible Kalman Filter Function)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = initial_p  
    q = q_val      
    r = r_val        
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

@st.cache_data(ttl=60)
def fetch_btc_fail_safe():
    binance_url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 720}
    try:
        response = requests.get(binance_url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            parsed_data = []
            for item in data:
                parsed_data.append({
                    "Timestamp": pd.to_datetime(item[0], unit='ms'),
                    "Open": float(item[1]),
                    "High": float(item[2]),
                    "Low": float(item[3]),
                    "Close": float(item[4])
                })
            df = pd.DataFrame(parsed_data)
            df.set_index("Timestamp", inplace=True)
            return df
    except Exception:
        pass 

    kraken_url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "XBTUSD", "interval": 60} 
    try:
        response = requests.get(kraken_url, params=params, timeout=5)
        if response.status_code == 200:
            res_json = response.json()
            data = res_json['result']['XXBTZUSD']
            parsed_data = []
            for item in data[-720:]:
                parsed_data.append({
                    "Timestamp": pd.to_datetime(item[0], unit='s'),
                    "Open": float(item[1]),
                    "High": float(item[2]),
                    "Low": float(item[3]),
                    "Close": float(item[4])
                })
            df = pd.DataFrame(parsed_data)
            df.set_index("Timestamp", inplace=True)
            return df
    except Exception:
        return pd.DataFrame()

with st.spinner("Processing Mathematical Breakout Matrix..."):
    df_raw = fetch_btc_fail_safe()
    if df_raw.empty:
        st.error("🚨 Connection issue. Please check your internet.")
        st.stop()
        
    df = df_raw.copy()

    # Base Matrix Definition
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0, q_val=0.001, r_val=0.1)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Pure Microstructure Probability Emulation
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(window=24).std() + 1e-10)
    df['Prob_Up'] = 1 / (1 + np.exp(-df['Normalized_Gap'])) # Math Sigmoid function replaces Sklearn ML
    df['Prob_Down'] = 1 - df['Prob_Up']

    # Arrays for loops
    prob_ups = df['Prob_Up'].to_numpy()
    prob_downs = df['Prob_Down'].to_numpy()
    closes = df['a_Close'].to_numpy()
    highs = df['High'].to_numpy()
    lows = df['Low'].to_numpy()
    kalmans_price = df['b_Kalman_Price'].to_numpy()

    final_signals, scores_log, raw_weighted_momentum_log = [], [], []
    accumulator = 0
    
    for i in range(len(prob_ups)):
        p_up, p_down, c_val, k_price_val = prob_ups[i], prob_downs[i], closes[i], kalmans_price[i]
        if p_up >= 0.55: accumulator += 1
        elif p_down >= 0.55: accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)
        raw_weighted_momentum_log.append(c_val - k_price_val)
        
        if accumulator == 5: final_signals.append("🟢 STRONG BUY (Max [5/5])")
        elif accumulator == -5: final_signals.append("🔴 STRONG SELL (Max [-5/-5])")
        else: final_signals.append(f"⚪ NEUTRAL/HOLD (Score: {accumulator})")

    df['d_ML_Signal'] = final_signals
    df['Accumulator_Score'] = scores_log  
    df['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 
    df['Weighted_Momentum'] = apply_kalman_filter_custom(df['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # =====================================================================
    # 💥 ZONE BREAKOUT LOGIC
    # =====================================================================
    w_moms = df['Weighted_Momentum'].to_numpy()
    buying_zone_low = lows[0]
    selling_zone_high = highs[0]
    zone_status_log = []
    breakout_signals = []

    for i in range(len(closes)):
        if w_moms[i] > 0:
            buying_zone_low = lows[i]  
        elif w_moms[i] < 0:
            selling_zone_high = highs[i] 

        zone_status_log.append(f"🟢 Buy:{buying_zone_low:.0f} | 🔴 Sell:{selling_zone_high:.0f}")

        if closes[i] > selling_zone_high and prob_ups[i] > 0.52:
            breakout_signals.append("💥 BREAKOUT BUY (Resistance Broken)")
        elif closes[i] < buying_zone_low and prob_downs[i] > 0.52:
            breakout_signals.append("💥 BREAKOUT SELL (Support Broken)")
        else:
            breakout_signals.append("⚖️ Inside Zone Tracking")

    df['Live_Tracked_Zones'] = zone_status_log
    df['Zone_Break_Signal'] = breakout_signals

    # UI Presentation
    clean_display_cols = ['a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Weighted_Momentum', 'Live_Tracked_Zones', 'Zone_Break_Signal']
    display_df = df[clean_display_cols].copy().iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live Breakout Dataset Matrix")
    st.dataframe(display_df, use_container_width=True, height=750)
