import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC Institutional Range Engine", layout="wide")
st.title("⚡ BTC-USD Live 1-Hour Standalone [Strict Live Flow Override]")
st.write("🎯 **Aapki Custom Setting:** Original W% Columns + **Pure High/Low 12-EMA Tunnel** + Signals Removed + Latest Candle Frozen on Top Row")

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

# -----------------------------------------------------------------
# 🛡️ ANTI-CRASH LIVE FETCH ENGINE WITH AUTOMATIC RECOVERY
# -----------------------------------------------------------------
df = None
is_simulated = False

with st.spinner("Executing Strict Live Data Fetch for BTC-USD (Max Safe Hourly Window)..."):
    current_time = datetime.now()
    start_date = current_time - timedelta(days=360) 
    end_date = current_time + timedelta(days=1) 
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    try:
        raw_df = yf.download("BTC-USD", start=start_str, end=end_str, interval="1h", progress=False)
        if len(raw_df) > 100:
            df = pd.DataFrame(index=raw_df.index)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in raw_df.columns:
                    df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]
            df.index = pd.to_datetime(df.index)
    except Exception as e:
        pass

    if df is None or len(df) < 100:
        is_simulated = True
        total_points = 4320
        np.random.seed(42)
        base_price = 1000.0
        price_path = [base_price]
        for i in range(1, total_points):
            cycle = np.sin(i / 150) * 4.0
            drift = 1.2 if cycle > 0 else -1.5
            noise = np.random.normal(0, 12)
            price_path.append(max(100, price_path[-1] + drift + noise))
            
        art_close = np.array(price_path)
        df = pd.DataFrame({
            'Open': art_close - np.random.uniform(-8, 8, size=total_points),
            'High': art_close + np.random.uniform(2, 15, size=total_points),
            'Low': art_close - np.random.uniform(2, 15, size=total_points),
            'Close': art_close,
            'Volume': [850000] * total_points
        }, index=pd.date_range(end=pd.Timestamp.now(), periods=total_points, freq='1h'))

if is_simulated:
    st.warning("⚠️ **YFinance API Call Restricted/Timed out.** App crash hone se bachane ke liye safe simulation mode auto-activate ho gaya hai!")
else:
    st.success("🟢 **Real Live Market Engine Running smoothly.**")

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

    # Live Accumulators & Raw Logs
    prob_up_vals = df_predict['Prob_Up_Raw'].to_numpy()
    prob_down_vals = df_predict['Prob_Down_Raw'].to_numpy()
    close_vals = df_predict['a_Close'].to_numpy()
    
    scores_log, raw_weighted_momentum_log = [], []
    accumulator = 0
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()

    for i in range(len(prob_up_vals)):
        p_up, p_down, c_val, k_price_val = prob_up_vals[i], prob_down_vals[i], close_vals[i], kalmans_price[i]
        if p_up >= 0.55: accumulator += 1
        elif p_down >= 0.55: accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)
        raw_weighted_momentum_log.append(c_val - k_price_val)

    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # Formatting UI Structure
    clean_display_cols = [
        'a_Close', 'b_Kalman_Price', 'Prob_Up_Raw', 'Prob_Down_Raw', 
        'KDiff_Prob_Up', 'KDiff_Prob_Down',
        'W_KalmanDiff(%)', 'W_OrderImb(%)', 'W_BodyImb(%)', 'W_NormGap(%)', 'W_Velocity(%)',
        'Accumulator_Score', 'Weighted_Momentum'
    ]
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Prob_Up_Raw'] = display_df['Prob_Up_Raw'].round(3)
    display_df['Prob_Down_Raw'] = display_df['Prob_Down_Raw'].round(3)
    display_df['KDiff_Prob_Up'] = display_df['KDiff_Prob_Up'].round(3)
    display_df['KDiff_Prob_Down'] = display_df['KDiff_Prob_Down'].round(3)
    
    for c in ['W_KalmanDiff(%)', 'W_OrderImb(%)', 'W_BodyImb(%)', 'W_NormGap(%)', 'W_Velocity(%)']:
        display_df[c] = display_df[c].round(1).astype(str) + "%"
        
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live Original Matrix + Pure Probability Streams (Signals Removed)")
    st.dataframe(display_df, use_container_width=True, height=750)
