import streamlit as st
import numpy as np
import pandas as pd
import requests

# Page Configuration
st.set_page_config(page_title="BTC Breakout Core Engine", layout="wide")
st.title("⚡ BTC Live 1-Hour Standalone Breakout Engine")
st.write("🎯 **Original Copy with Structural Breakout Engine:** 50:50 Split + Binance Node + **Dynamic Buying/Selling Zone Breakout Tracker Added**")

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
def fetch_btc_from_binance():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 750}
    try:
        response = requests.get(url, params=params, timeout=10)
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
        return pd.DataFrame()

with st.spinner("Compiling Breakout Blocks & Loading Core Matrix..."):
    df_raw = fetch_btc_from_binance()
    if df_raw.empty:
        st.error("Binance Node Server Timeout. Please refresh.")
        st.stop()
        
    df = df_raw.copy()

    # Base Matrix Definition (Price Kalman 1 Active)
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0, q_val=0.001, r_val=0.1)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Microstructure Features Space
    df['Sign_Change'] = (np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))).astype(int)
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(window=24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Target Rule
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix + ['Target'], inplace=True)

# Dynamic Split Engine (Strict 50:50 Ratio)
split_idx = int(len(df) * 0.50)
df_train = df.iloc[:split_idx]
X_train = df_train[features_matrix].copy()
y_train = df_train['Target'].copy()

df_predict = df.iloc[split_idx:].copy()
X_predict = df_predict[features_matrix].copy()

if len(X_predict) != 0:
    model_flow = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42)
    model_flow.fit(X_train, y_train)

    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # Arrays for loops
    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    highs = df_predict['High'].to_numpy()
    lows = df_predict['Low'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()

    final_signals, scores_log, raw_weighted_momentum_log = [], [], []
    accumulator = 0
    
    # Pre-calculating original baseline values
    for i in range(len(prob_ups)):
        p_up, p_down, c_val, k_price_val = prob_ups[i], prob_downs[i], closes[i], kalmans_price[i]
        if p_up >= 0.55: accumulator += 1
        elif p_down >= 0.55: accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)
        raw_weighted_momentum_log.append(c_val - k_price_val)
        
        if accumulator == 5: final_signals.append("🟢 STRONG BUY (Max Locked [5/5])")
        elif accumulator == -5: final_signals.append("🔴 STRONG SELL (Max Locked [-5/-5])")
        else: final_signals.append(f"⚪ NEUTRAL/HOLD (Score: {accumulator})")

    df_predict['d_ML_Signal'] = final_signals
    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # =====================================================================
    # 💥 AUTOMATED ORDER BLOCK & ZONE BREAKOUT BACKEND LOGIC
    # =====================================================================
    w_moms = df_predict['Weighted_Momentum'].to_numpy()
    
    buying_zone_low = lows[0]
    selling_zone_high = highs[0]
    zone_status_log = []
    breakout_signals = []

    for i in range(len(closes)):
        # Dynamic Zone Assignment based on underlying Kalman Momentum
        if w_moms[i] > 0:
            buying_zone_low = lows[i]  # New Support Zone Established
        elif w_moms[i] < 0:
            selling_zone_high = highs[i] # New Resistance Zone Established

        zone_status_log.append(f"🟢 Buy:{buying_zone_low:.0f} | 🔴 Sell:{selling_zone_high:.0f}")

        # Breakout Logic Identification Engine
        # 1. Bullish Breakout: If close breaks above the tracked selling zone boundary
        if closes[i] > selling_zone_high and prob_ups[i] > 0.52:
            breakout_signals.append("💥 BREAKOUT BUY (Resistance Broken)")
        # 2. Bearish Breakout: If close breaks below the tracked buying zone boundary
        elif closes[i] < buying_zone_low and prob_downs[i] > 0.52:
            breakout_signals.append("💥 BREAKOUT SELL (Support Broken)")
        else:
            breakout_signals.append("⚖️ Inside Zone Tracking")

    df_predict['Live_Tracked_Zones'] = zone_status_log
    df_predict['Zone_Break_Signal'] = breakout_signals

    # Formatting UI Structure
    clean_display_cols = ['a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Weighted_Momentum', 'Live_Tracked_Zones', 'Zone_Break_Signal']
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    
    # Flip index to have the newest candle on top row
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live Breakout Dataset Matrix (Latest Hour Locked on Top Row)")
    st.dataframe(display_df, use_container_width=True, height=750)
