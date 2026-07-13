import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC Institutional Range Engine", layout="wide")
st.title("⚡ BTC-USD Live 1-Hour Standalone [Strict Live Flow Override]")

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
# 🛡️ BULLET-PROOF DATA ENGINE WITH AUTOMATIC SIMULATION FALLBACK
# -----------------------------------------------------------------
df = None
is_fallback_active = False

try:
    current_time = datetime.now()
    start_date = current_time - timedelta(days=180) 
    end_date = current_time + timedelta(days=1) 
    
    # Attempting real live market fetch
    raw_df = yf.download("BTC-USD", start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1h", progress=False)
    
    if len(raw_df) > 100:
        df = pd.DataFrame(index=raw_df.index)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in raw_df.columns:
                df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]
        df.index = pd.to_datetime(df.index)
except Exception as live_err:
    pass

# If live API fails, activate the 4,320 Rows Fail-Safe Stream instantly
if df is None or len(df) < 100:
    is_fallback_active = True
    total_points = 4320
    np.random.seed(42)
    
    base_price = 1000.0
    price_path = [base_price]
    for i in range(1, total_points):
        cycle = np.sin(i / 150) * 4.0
        if 2500 <= i <= 3100:
            drift = 0.0
            noise = np.random.normal(0, 3) 
        else:
            drift = 1.2 if cycle > 0 else -1.5
            noise = np.random.normal(0, 14) 
        price_path.append(max(100, price_path[-1] + drift + noise))
        
    art_close = np.array(price_path)
    df = pd.DataFrame({
        'Open': art_close - np.random.uniform(-8, 8, size=total_points),
        'High': art_close + np.random.uniform(2, 16, size=total_points),
        'Low': art_close - np.random.uniform(2, 16, size=total_points),
        'Close': art_close,
        'Volume': [850000] * total_points
    }, index=pd.date_range(end=pd.Timestamp.now(), periods=total_points, freq='1h'))

# Status Display Banner
if is_fallback_active:
    st.warning("⚠️ **Yahoo Finance Live API Blocked/Offline!** Dashboard ko breakdown se bachane ke liye **Fail-Safe Internal Simulated Stream (₹1000 Base, 4,320 Hourly Rows)** ko automatic inject kar diya gaya hai. Aap calculations test kar sakte hain.")
else:
    st.success("🟢 **Real-Time Live Market Data Stream Connected Successfully!**")

st.write("🎯 **Aapki Custom Setting:** GitHub Format Locked + Target WMA High/Low Prices + Strict 12-WMA Linear Tunnel + Latest Row Frozen on Top")

# Base Matrix Calculations (GitHub Format Intact)
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
    st.error("Prediction matrix boundary error.")
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
    df_predict['W_KalmanDiff(%)'] = feat_weights_arr[:, 0]
    df_predict['W_OrderImb(%)'] = feat_weights_arr[:, 1]
    df_predict['W_BodyImb(%)'] = feat_weights_arr[:, 2]
    df_predict['W_NormGap(%)'] = feat_weights_arr[:, 3]
    df_predict['W_Velocity(%)'] = feat_weights_arr[:, 4]

    # -----------------------------------------------------------------
    # 🧠 INTEGRATION: STRICT 12-WMA TUNNEL ENGINE (Weights: 12,11,10...1)
    # -----------------------------------------------------------------
    wma_weights = np.arange(12, 0, -1)
    wma_sum = np.sum(wma_weights)

    def calculate_pure_wma(series):
        return series.rolling(window=12).apply(lambda x: np.sum(x * wma_weights) / wma_sum, raw=True)

    df['WMA_High_Tunnel'] = calculate_pure_wma(df['High'])
    df['WMA_Low_Tunnel'] = calculate_pure_wma(df['Low'])

    df_predict['Target_WMA_High'] = df['WMA_High_Tunnel'].loc[df_predict.index]
    df_predict['Target_WMA_Low'] = df['WMA_Low_Tunnel'].loc[df_predict.index]

    prob_up_vals = df_predict['Prob_Up_Raw'].to_numpy()
    prob_down_vals = df_predict['Prob_Down_Raw'].to_numpy()
    close_vals = df_predict['a_Close'].to_numpy()
    high_tunnel_vals = df_predict['Target_WMA_High'].to_numpy()
    low_tunnel_vals = df_predict['Target_WMA_Low'].to_numpy()
    
    final_prob_up_ui = []
    final_prob_down_ui = []

    for idx in range(len(close_vals)):
        p_up = prob_up_vals[idx]
        p_down = prob_down_vals[idx]
        current_close = close_vals[idx]
        current_high_band = high_tunnel_vals[idx]
        current_low_band = low_tunnel_vals[idx]

        if np.isnan(current_high_band) or np.isnan(current_low_band):
            final_prob_up_ui.append("🔄 LOADING")
            final_prob_down_ui.append("🔄 LOADING")
            continue

        if current_close > current_high_band:
            final_prob_up_ui.append("🟢 BUY SIGNAL")
            final_prob_down_ui.append(str(round(p_down, 3)))
        elif current_close < current_low_band:
            final_prob_up_ui.append(str(round(p_up, 3)))
            final_prob_down_ui.append("🔴 SELL SIGNAL")
        else:
            final_prob_up_ui.append("⏳ TRAP ZONE")
            final_prob_down_ui.append("⏳ TRAP ZONE")

    df_predict['Prob_Up'] = final_prob_up_ui
    df_predict['Prob_Down'] = final_prob_down_ui

    # Live Accumulators & Raw Logs
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

    # UI Construction (Exact Template Replicated)
    clean_display_cols = [
        'a_Close', 'Target_WMA_High', 'Target_WMA_Low', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 
        'KDiff_Prob_Up', 'KDiff_Prob_Down',
        'W_KalmanDiff(%)', 'W_OrderImb(%)', 'W_BodyImb(%)', 'W_NormGap(%)', 'W_Velocity(%)',
        'Accumulator_Score', 'Weighted_Momentum'
    ]
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['Target_WMA_High'] = display_df['Target_WMA_High'].round(2)
    display_df['Target_WMA_Low'] = display_df['Target_WMA_Low'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['KDiff_Prob_Up'] = display_df['KDiff_Prob_Up'].round(3)
    display_df['KDiff_Prob_Down'] = display_df['KDiff_Prob_Down'].round(3)
    
    for c in ['W_KalmanDiff(%)', 'W_OrderImb(%)', 'W_BodyImb(%)', 'W_NormGap(%)', 'W_Velocity(%)']:
        display_df[c] = display_df[c].astype(str).apply(lambda x: x.split('.')[0] + '%')
        
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Operational Dashboard Matrix (Active Data Rows: {len(display_df)})")
    st.dataframe(display_df, use_container_width=True, height=750)
