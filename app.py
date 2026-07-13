import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC Institutional Range Engine", layout="wide")
st.title("⚡ BTC-USD Live 1-Hour Standalone [Strict Live Flow Override]")
st.write("🎯 **Aapki Custom Setting:** Fast + Slow Dual Kalman Filter Synchronization Matrix + VIDYA Accumulator Engine + Latest Candle Frozen on Top Row")

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
# 🛡️ ANTI-CRASH LIVE FETCH ENGINE WITH AUTOMATIC RECOVERY
# -----------------------------------------------------------------
df = None
is_simulated = False

with st.spinner("Executing Strict Live Data Fetch for BTC-USD..."):
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
    st.warning("⚠️ **YFinance API Call Restricted/Timed out.** Safe simulation mode auto-activated.")
else:
    st.success("🟢 **Real Live Market Engine Running smoothly.**")

# Base Matrix Definition
df['a_Close'] = df['Close']

# DUAL KALMAN GENERATION ON CLOSE PRICE (Fast vs Slow Engine)
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
    df_predict['W_NormGap(%)_Raw'] = feat_weights_arr[:, 3]
    df_predict['W_Velocity(%)_Raw'] = feat_weights_arr[:, 4]

    # -----------------------------------------------------------------
    # 🧠 DUAL KALMAN TUNNEL & STRICT DUAL ALIGNMENT CROSSOVER LOGIC
    # -----------------------------------------------------------------
    wma_weights = np.arange(12, 0, -1) 
    wma_sum = np.sum(wma_weights)       

    # Calculate 12-WMA Tunnels for both Kalman lines
    df['Fast_WMA_Tunnel'] = df['b_Kalman_Price'].rolling(window=12).apply(lambda x: np.sum(x * wma_weights) / wma_sum, raw=True)
    df['Slow_WMA_Tunnel'] = df['Slow_Kalman_Price'].rolling(window=12).apply(lambda x: np.sum(x * wma_weights) / wma_sum, raw=True)
    
    df_predict['Slow_Kalman_Price'] = df['Slow_Kalman_Price'].loc[df_predict.index]
    df_predict['Fast_WMA_Tunnel'] = df['Fast_WMA_Tunnel'].loc[df_predict.index]
    df_predict['Slow_WMA_Tunnel'] = df['Slow_WMA_Tunnel'].loc[df_predict.index]

    fast_vals = df_predict['b_Kalman_Price'].to_numpy()
    slow_vals = df_predict['Slow_Kalman_Price'].to_numpy()
    fast_wma = df_predict['Fast_WMA_Tunnel'].to_numpy()
    slow_wma = df_predict['Slow_WMA_Tunnel'].to_numpy()
    
    signal_log = []
    for idx in range(len(fast_vals)):
        if np.isnan(fast_wma[idx]) or np.isnan(slow_wma[idx]):
            signal_log.append("⏳ LOADING")
            continue
            
        fast_bullish = fast_vals[idx] > fast_wma[idx]
        slow_bullish = slow_vals[idx] > slow_wma[idx]
        
        fast_bearish = fast_vals[idx] < fast_wma[idx]
        slow_bearish = slow_vals[idx] < slow_wma[idx]
        
        # Dual Confirmation Filter Strategy
        if fast_bullish and slow_bullish:
            signal_log.append("🟢 BUY")
        elif fast_bearish and slow_bearish:
            signal_log.append("🔴 SELL")
        else:
            signal_log.append("⏳ WAIT ZONE")

    df_predict['Signal'] = signal_log

    # -----------------------------------------------------------------
    # 🧠 VIDYA INDICATOR & ACCUMULATOR MATRICES
    # -----------------------------------------------------------------
    df['Vidhya'] = apply_vidya_custom(df['a_Close'].values, period=14)
    df['Close_Minus_Vidhya'] = df['a_Close'] - df['Vidhya']
    df['VIDYA_Weighted_Momentum'] = apply_kalman_filter_custom(df['Close_Minus_Vidhya'].values, initial_p=0.50, q_val=0.001, r_val=0.1)
    
    vidya_mom_vals = df['VIDYA_Weighted_Momentum'].values
    vidya_accum_log = np.zeros_like(vidya_mom_vals)
    v_accum = 0
    for idx in range(1, len(vidya_mom_vals)):
        if vidya_mom_vals[idx] > vidya_mom_vals[idx-1]: v_accum += 1
        elif vidya_mom_vals[idx] < vidya_mom_vals[idx-1]: v_accum -= 1
        v_accum = max(-5, min(5, v_accum))
        vidya_accum_log[idx] = v_accum
    df['VIDYA_Accumulator_Score'] = vidya_accum_log

    df_predict['Vidhya'] = df['Vidhya'].loc[df_predict.index]
    df_predict['Close_Minus_Vidhya'] = df['Close_Minus_Vidhya'].loc[df_predict.index]
    df_predict['VIDYA_Weighted_Momentum'] = df['VIDYA_Weighted_Momentum'].loc[df_predict.index]
    df_predict['VIDYA_Accumulator_Score'] = df['VIDYA_Accumulator_Score'].loc[df_predict.index].astype(int)

    # Core Live Accumulators 
    prob_up_vals = df_predict['Prob_Up_Raw'].to_numpy()
    prob_down_vals = df_predict['Prob_Down_Raw'].to_numpy()
    close_vals = df_predict['a_Close'].to_numpy()
    
    scores_log, raw_weighted_momentum_log = [], []
    accumulator = 0
    for i in range(len(prob_up_vals)):
        p_up, p_down = prob_up_vals[i], prob_down_vals[i]
        if p_up >= 0.55: accumulator += 1
        elif p_down >= 0.55: accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)
        raw_weighted_momentum_log.append(close_vals[i] - fast_vals[i])

    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum_log, initial_p=0.50, q_val=0.001, r_val=0.1)

    # UI Conversion
    df_predict['W_KalmanDiff(%)'] = df_predict['W_KalmanDiff(%)_Raw'].round(1).astype(str) + "%"
    df_predict['W_OrderImb(%)'] = df_predict['W_OrderImb(%)_Raw'].round(1).astype(str) + "%"
    df_predict['W_BodyImb(%)'] = df_predict['W_BodyImb(%)_Raw'].round(1).astype(str) + "%"
    df_predict['W_NormGap(%)'] = df_predict['W_NormGap(%)_Raw'].round(1).astype(str) + "%"
    df_predict['W_Velocity(%)'] = df_predict['W_Velocity(%)_Raw'].round(1).astype(str) + "%"

    # Sequential UI Columns Definition Matrix (Strict Ordered Lock)
    clean_display_cols = [
        'a_Close', 'Vidhya', 'Close_Minus_Vidhya', 'VIDYA_Weighted_Momentum', 'VIDYA_Accumulator_Score',
        'b_Kalman_Price', 'Fast_WMA_Tunnel', 'Slow_Kalman_Price', 'Slow_WMA_Tunnel', 'Signal', 
        'Prob_Up_Raw', 'Prob_Down_Raw', 'KDiff_Prob_Up', 'KDiff_Prob_Down',
        'W_KalmanDiff(%)', 'W_OrderImb(%)', 'W_BodyImb(%)', 'W_NormGap(%)', 'W_Velocity(%)',
        'Accumulator_Score', 'Weighted_Momentum'
    ]
    display_df = df_predict[clean_display_cols].copy()
    
    for c in ['a_Close', 'Vidhya', 'Close_Minus_Vidhya', 'VIDYA_Weighted_Momentum', 'b_Kalman_Price', 'Fast_WMA_Tunnel', 'Slow_Kalman_Price', 'Slow_WMA_Tunnel', 'Weighted_Momentum']:
        display_df[c] = display_df[c].round(2)
    for c in ['Prob_Up_Raw', 'Prob_Down_Raw', 'KDiff_Prob_Up', 'KDiff_Prob_Down']:
        display_df[c] = display_df[c].round(3)
        
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live Original Matrix + Synchronized Fast/Slow Kalman Anti-Whipsaw Filter Matrix Active")
    st.dataframe(display_df, use_container_width=True, height=750)
