import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import pytz

# Page Configuration
st.set_page_config(page_title="BTC Tree Strike Scoreboard", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Hour [Umesh Scoreboard Edition]")
st.write("🎯 **ML Decision Tree Filter:** Tracks 0.10 -> 0.88 jumps, calculates dynamic strikes, and updates the live scoreboard on the left panel.")

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

with st.spinner("Synchronizing Data Streams & Scoreboard Engines..."):
    IST = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(IST)
    
    start_date = current_time - timedelta(days=720) 
    end_date = current_time + timedelta(days=1) 
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    raw_df = yf.download("BTC-USD", start=start_str, end=end_str, interval="1h")
    
    if len(raw_df) == 0:
        st.error(f"YFinance Connection Timeout. Please refresh.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df.index = pd.to_datetime(df.index)

    # Base Matrix Definition
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
    
    df['State_Direction'] = np.where(df['c_Combined'] > 0, 1, 0)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix + ['State_Direction'], inplace=True)

# Dynamic Split Engine (Strict 50:50 Ratio)
split_idx = int(len(df) * 0.50)
df_train = df.iloc[:split_idx]
X_train = df_train[features_matrix].copy()
y_train = df_train['State_Direction'].copy()

df_predict = df.iloc[split_idx:].copy()
X_predict = df_predict[features_matrix].copy()

if len(X_predict) == 0:
    st.error("Prediction matrix error.")
else:
    model_flow = RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=1, random_state=42)
    model_flow.fit(X_train, y_train)

    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # Live Accumulators & Raw Logs
    scores_log, raw_weighted_momentum_log = [], []
    accumulator = 0
    
    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()

    for i in range(len(prob_ups)):
        p_up, p_down, c_val, k_price_val = prob_ups[i], prob_downs[i], closes[i], kalmans_price[i]
        if p_up >= 0.55: accumulator += 1
        elif p_down >= 0.55: accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)
        raw_weighted_momentum_log.append(c_val - k_price_val)

    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # -----------------------------------------------------------------
    # 🎯 CORE LOGIC & LIVE TRACKING SCOREBOARD
    # -----------------------------------------------------------------
    tree_strike_logs = []
    current_position = None  
    decision_trees = model_flow.estimators_
    
    # Trackers for Scoreboard
    total_signals = 0
    sahi_trades = 0
    galat_trades = 0
    
    for i in range(len(df_predict)):
        if i == 0:
            tree_strike_logs.append("-")
