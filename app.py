import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC Institutional Range Engine", layout="wide")
st.title("⚡ BTC-USD Live 1-Hour Standalone [Strict Live Flow Override]")
st.write("🎯 **Aapki Custom Setting:** Original W% Columns + **Inverted Weighted 25-EMA Tunnel (Max Weight on 25)** + Latest Candle Frozen on Top Row")

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

with st.spinner("Executing Strict Live Data Fetch for BTC-USD..."):
    current_time = datetime.now()
    start_date = current_time - timedelta(days=720) 
    end_date = current_time + timedelta(days=1) 
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    raw_df = yf.download("BTC-USD", start=start_str, end=end_str, interval="1h")
    
    if len(raw_df) == 0:
        st.error(f"YFinance API Limit Error. Please refresh.")
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
    # Model 1: Core 5-Feature Model
    model_flow = RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=1, random_state=42)
    model_flow.fit(X_train, y_train)

    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down_Raw'] = probabilities[:, 0]
    df_predict['Prob_Up_Raw'] = probabilities[:, 1]

    # Model 2: Isolated Kalman Diff Model
    X_train_kdiff = df_train[['c_Combined']].copy()
    X_predict_kdiff = df_predict[['c_Combined']].copy()
    model_kdiff = RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=1, random_state=42)
    model_kdiff.fit(X_train_kdiff, y_train)
    prob_kdiff = model_kdiff.predict_proba(X_predict_kdiff)
    df_predict['KDiff_Prob_Up'] = prob_kdiff[:, 1]
    df_predict['KDiff_Prob_Down'] = prob_kdiff[:, 0]

    # Feature Importance / Dominance Math (W%)
    importances = model_flow.feature_importances_
    feat_weights = []
    X_predict_arr = X_predict.to_numpy()
    X_train_mean = X_train.mean().to_numpy()
    X_train_std = X_train.std().to_numpy() + 1e-10

    for row in X_predict_arr:
        deviation = np.abs(row - X_train_mean) / X_train_std
        raw_contrib = deviation * importances
        total_contrib = np.sum(raw_contrib) + 1e-10
        feat_weights.append((raw_contrib / total_contrib) * 100)

    feat_weights_arr = np.array(feat_weights)
    df_predict['W_KalmanDiff(%)'] = feat_weights_arr[:, 0]
    df_predict['W_OrderImb(%)'] = feat_weights_arr[:, 1]
    df_predict['W_BodyImb(%)'] = feat_weights_arr[:, 2]
    df_predict['W_NormGap(%)'] = feat_weights_arr[:, 3]
    df_predict['W_Velocity(%)'] = feat_weights_arr[:, 4]

    # -----------------------------------------------------------------
    # 🧠 INVERTED WEIGHTED EMA TUNNEL ENGINE (Max Weight on EMA 25)
    # -----------------------------------------------------------------
    # Inverting the weights: EMA 25 gets weight 25, EMA 24 gets 24 ... EMA 1 gets weight 1
    ema_lengths = np.arange(1, 26)
    inverted_decay_weights = ema_lengths  # [1, 2, 3, ..., 25] -> Sabse akhri (25) ko sabse jada weight
    total_weight_sum = np.sum(inverted_decay_weights)

    high_ema_stack = []
    low_ema_stack = []

    for length in ema_lengths:
        high_ema_stack.append(df['High'].ewm(span=int(length), adjust=False).mean())
        low_ema_stack.append(df['Low'].ewm(span=int(length), adjust=False).mean())

    weighted_high_tunnel = np.zeros(len(df))
    weighted_low_tunnel = np.zeros(len(df))

    for i in range(25):
        weighted_high_tunnel += high_ema_stack[i].to_numpy() * (inverted_decay_weights[i] / total_weight_sum)
        weighted_low_tunnel += low_ema_stack[i].to_numpy() * (inverted_decay_weights[i] / total_weight_sum)

    df['Weighted_High_Tunnel'] = weighted_high_tunnel
    df['Weighted_Low_Tunnel'] = weighted_low_tunnel

    df_predict['Weighted_High_Tunnel'] = df['Weighted_High_Tunnel'].loc[df_predict.index]
    df_predict['Weighted_Low_Tunnel'] = df['Weighted_Low_Tunnel'].loc[df_predict.index]

    prob_up_vals = df_predict['Prob_Up_Raw'].to_numpy()
    prob_down_vals = df_predict['Prob_Down_Raw'].to_numpy()
    close_vals = df_predict['a_Close'].to_numpy()
    high_tunnel_vals = df_predict['Weighted_High_Tunnel'].to_numpy()
    low_tunnel_vals = df_predict['Weighted_Low_Tunnel'].to_numpy()
    
    final_prob_up_ui = []
    final_prob_down_ui = []

    for idx in range(len(close_vals)):
        p_up = prob_up_vals[idx]
        p_down = prob_down_vals[idx]
        current_close = close_vals[idx]
        current_high_band = high_tunnel_vals[idx]
        current_low_band = low_tunnel_vals[idx]

        # Cross Execution Logic based on Inverted Deep Tunnel
        if current_close > current_high_band:
            final_prob_up_ui.append("🟢 BUY SIGNAL")
            final_prob_down_ui.append(str(round(p_down, 3)))
        elif current_close < current_low_band:
            final_prob_up_ui.append(str(round(p_up, 3)))
            final_prob_down_ui.append("🔴 SELL SIGNAL")
        else:
            final_prob_up_ui.append("⏳ TRAP ZONE")
            final_prob_down_ui.append("⏳ TRAP ZONE")

    df_predict
