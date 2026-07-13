import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC Institutional Range Engine", layout="wide")
st.title("⚡ BTC-USD Live 1-Hour Standalone [Strict Live Flow Override]")
st.write("🎯 **Aapki Favorite Setting:** Strict 6-Month Period Loop + Double Kalman Filter Confirmation Matrix + VIDYA Accumulator + Latest Candle Frozen on Top")

# =====================================================================
# MATHEMATICAL ENGINE (Flexible Kalman Filter & VIDYA Functions)
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

def apply_vidya_custom(data_array, period=14):
    if len(data_array) < period:
        return data_array.copy()
        
    s = pd.Series(data_array)
    diff = s.diff()
    gains = diff.where(diff > 0, 0)
    losses = (-diff).where(diff < 0, 0)
    
    sum_gains = gains.rolling(window=period).sum()
    sum_losses = losses.rolling(window=period).sum()
    
    cmo = (sum_gains - sum_losses) / (sum_gains + sum_losses + 1e-10)
    k = cmo.abs().fillna(0).to_numpy()
    
    alpha = 2 / (period + 1)
    vidya_values = np.zeros_like(data_array)
    vidya_values[0] = data_array[0]
    
    for i in range(1, len(data_array)):
        vidya_values[i] = (alpha * k[i] * data_array[i]) + (1 - alpha * k[i]) * vidya_values[i-1]
        
    return vidya_values

# -----------------------------------------------------------------
# 🛡️ ANTI-CRASH LIVE YAHOO FINANCE DATA ENGINE (STRICT 6-MONTH)
# -----------------------------------------------------------------
df = None
is_simulated = False

with st.spinner("Executing Strict 6-Month Yahoo Finance Data Fetch..."):
    try:
        raw_df = yf.download("BTC-USD", period="6m", interval="1h", progress=False)
        if raw_df is not None and len(raw_df) > 100:
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
    st.warning("⚠️ **Yahoo Server pipe restricted.** Safe simulation mode auto-activated.")
else:
    st.success("🟢 **Real Live Market Engine Running smoothly (Strict 6-Month Loop Active).**")

# Base Matrix Definition
df['a_Close'] = df['Close']

# DUAL INSTITUTIONAL KALMAN ENGINE GENERATION
df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0, q_val=0.001, r_val=0.1)
df['Slow_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0, q_val=0.00001, r_val=0.9)

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

# Dynamic Split Engine (Strict 50:50 Ratio on 6-Month Data Window)
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

    # Feature Importance Math (W%)
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
    df_predict['W_KalmanDiff(%)_Raw'] = feat_weights_arr[:, 0]
    df_predict['W_OrderImb(%)_Raw'] = feat_weights_arr[:, 1]
    df_predict['W_BodyImb(%)_Raw'] = feat_weights_arr[:, 2]
    df_predict['W_NormGap(%)_Raw'] =
