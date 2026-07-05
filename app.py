import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="BTC Clean Kalman Engine", layout="wide")
st.title("⚡ BTC Live 1-Hour Standalone Breakout Engine")
st.write("🎯 **Clean Setup:** 2-Year Window + Kalman 3 Shifted to Momentum Difference Raw")

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
def pull_historical_data_failsafe():
    """Dual-Node Network: Pulls optimized historical data from YFinance with automatic Kraken fallback"""
    requested_period = "730d" 
    try:
        raw_df = yf.download("BTC-USD", period=requested_period, interval="1h", multi_level_index=False)
        if not raw_df.empty and len(raw_df) > 500:
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = [str(col[0]).upper() for col in raw_df.columns]
            else:
                raw_df.columns = [str(col).upper() for col in raw_df.columns]
                
            df = pd.DataFrame(index=raw_df.index)
            df['Open'] = pd.to_numeric(raw_df['OPEN'].values.flatten(), errors='coerce')
            df['High'] = pd.to_numeric(raw_df['HIGH'].values.flatten(), errors='coerce')
            df['Low'] = pd.to_numeric(raw_df['LOW'].values.flatten(), errors='coerce')
            df['Close'] = pd.to_numeric(raw_df['CLOSE'].values.flatten(), errors='coerce')
            df['Volume'] = pd.to_numeric(raw_df['VOLUME'].values.flatten(), errors='coerce')
            return df
    except Exception:
        pass

    try:
        kraken_url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": "XBTUSD", "interval": 60} 
        response = requests.get(kraken_url, params=params, timeout=10)
        if response.status_code == 200:
            res_json = response.json()
            data = res_json['result']['XXBTZUSD']
            parsed_data = []
            for item in data:
                parsed_data.append({
                    "Timestamp": pd.to_datetime(item[0], unit='s'),
                    "Open": float(item[1]),
                    "High": float(item[2]),
                    "Low": float(item[3]),
                    "Close": float(item[4]),
                    "Volume": float(item[6])
                })
            df = pd.DataFrame(parsed_data)
            df.set_index("Timestamp", inplace=True)
            return df
    except Exception:
        return pd.DataFrame()

with st.spinner("Processing Matrix Framework..."):
    df_raw = pull_historical_data_failsafe()
    if df_raw.empty:
        st.error("🚨 Both Data Endpoints are unreachable or returned empty frames. Please refresh.")
        st.stop()
        
    df = df_raw.copy()
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # VOLUME ENGINE: Dynamic Multiplier (24 Hour Baseline Setup)
    df['Vol_MA_24'] = df['Volume'].rolling(window=24).mean()
    df['Vol_Multiplier'] = df['Volume'] / (df['Vol_MA_24'] + 1e-10)
    df['Vol_Multiplier'] = df['Vol_Multiplier'].clip(lower=0.5, upper=3.0)

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
    df.dropna(subset=features_matrix + ['Target', 'Vol_Multiplier'], inplace=True)

# EXACT 50:50 MATRIX RATIO SPLIT
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

    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()
    vol_mults = df_predict['Vol_Multiplier'].to_numpy()

    scores_log, raw_weighted_momentum_log = [], []
    accumulator = 0
    
    for i in range(len(prob_ups)):
        p_up, p_down, c_val, k_price_val = prob_ups[i], prob_downs[i], closes[i], kalmans_price[i]
        if p_up >= 0.55: accumulator += 1
        elif p_down >= 0.55: accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)
        
        raw_weighted_momentum_log.append(c_val - k_price_val)

    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 
    
    # 1. Kalman 2: Standard Price-Based Weighted Momentum
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)
    
    # 2. Volume Multiplied Momentum Layer (Raw)
    df_predict['Vol_Multiplied_Momentum'] = df_predict['Weighted_Momentum'] * vol_mults
    
    # 3. Difference between Weighted Momentum and Volume Multiplied Momentum
    df_predict['Momentum_Diff_Raw'] = df_predict['Weighted_Momentum'] - df_predict['Vol_Multiplied_Momentum']
    
    # 🔥 MODIFIED: Kalman 3 is now calculated directly on Momentum_Diff_Raw
    df_predict['Kalman_Vol_Momentum'] = apply_kalman_filter_custom(df_predict['Momentum_Diff_Raw'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # Dynamic Signal Engine (Baaki sab copy paste)
    final_signals = []
    wm_vals = df_predict['Weighted_Momentum'].to_numpy()
    k3_vals = df_predict['Kalman_Vol_Momentum'].to_numpy()

    for i in range(len(wm_vals)):
        diff = wm_vals[i] - k3_vals[i]
        if diff > 0:
            final_signals.append(f"🟢 STRONG BUY (Diff: +{round(diff, 2)})")
        else:
            final_signals.append(f"🔴 STRONG SELL (Diff: {round(diff, 2)})")

    df_predict['d_ML_Signal'] = final_signals

    # Formatting Clean Output Frame
    clean_display_cols = [
        'a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 
        'Weighted_Momentum', 'Vol_Multiplied_Momentum', 'Momentum_Diff_Raw', 
        'Kalman_Vol_Momentum', 'd_ML_Signal'
    ]
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    display_df['Vol_Multiplied_Momentum'] = display_df['Vol_Multiplied_Momentum'].round(2) 
    display_df['Momentum_Diff_Raw'] = display_df['Momentum_Diff_Raw'].round(2) 
    display_df['Kalman_Vol_Momentum'] = display_df['Kalman_Vol_Momentum'].round(2) 
    
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live Volumetric Kalman Matrix Frame (Latest Hour Active on Top Row)")
    st.dataframe(display_df, use_container_width=True, height=750)
