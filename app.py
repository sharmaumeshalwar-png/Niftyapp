import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC 2-Year Institutional Range Engine", layout="wide")
st.title("⚡ BTC-USD Live 2-Year 1-Hour Engine [Pagination Matrix]")
st.write("🎯 **Historical Scale:** Pure 2-Year 1-Hour Stream via Paginated Binance Pipe + Strict Price-Kalman Guardrails")

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
# 🛡️ PAGINATED BINANCE PUBLIC API ENGINE (FETCHES PURE 2-YEAR DATA)
# -----------------------------------------------------------------
df = None
is_simulated = False

with st.spinner("Compiling 2 Years of 1-Hour Ticks via Paginated Binance Stream..."):
    try:
        all_candles = []
        # Calculate timestamps for 2 years ago
        end_time = int(time.time() * 1000)
        start_target = int((datetime.now() - timedelta(days=730)).timestamp() * 1000)
        
        current_end = end_time
        # Loop to fetch 1000 candles at a time until 2 years are covered
        while current_end > start_target:
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000&endTime={current_end}"
            response = requests.get(url, timeout=15)
            data = response.json()
            
            if not isinstance(data, list) or len(data) == 0:
                break
                
            all_candles = data + all_candles  # Prepend historical batch
            # Update end time to the earliest candle's timestamp minus 1 millisecond
            first_candle_time = data[0][0]
            if current_end == first_candle_time:
                break
            current_end = first_candle_time - 1
            
            # Safety break to comply with rate limits
            time.sleep(0.1)
            if len(all_candles) >= 18000:  # Cap at 2-year equivalent rows
                break

        if len(all_candles) > 500:
            parsed_data = []
            for item in all_candles:
                parsed_data.append({
                    'Timestamp': pd.to_datetime(item[0], unit='ms'),
                    'Open': float(item[1]),
                    'High': float(item[2]),
                    'Low': float(item[3]),
                    'Close': float(item[4]),
                    'Volume': float(item[5])
                })
            df = pd.DataFrame(parsed_data)
            df.drop_duplicates(subset=['Timestamp'], inplace=True)
            df.set_index('Timestamp', inplace=True)
    except Exception as e:
        pass

    # Safety Simulator Fallback Mode (Generates 17,500 rows if APIs drop)
    if df is None or len(df) < 500:
        is_simulated = True
        total_points = 17520  # 24 hours * 730 days
        np.random.seed(42)
        base_price = 63000.0
        price_path = [base_price]
        for i in range(1, total_points):
            cycle = np.sin(i / 300) * 5.0
            drift = 1.1 if cycle > 0 else -1.3
            noise = np.random.normal(0, 15)
            price_path.append(max(10000, price_path[-1] + drift + noise))
            
        art_close = np.array(price_path)
        df = pd.DataFrame({
            'Open': art_close - np.random.uniform(-10, 10, size=total_points),
            'High': art_close + np.random.uniform(2, 20, size=total_points),
            'Low': art_close - np.random.uniform(2, 20, size=total_points),
            'Close': art_close,
            'Volume': [950000] * total_points
        }, index=pd.date_range(end=pd.Timestamp.now(), periods=total_points, freq='1h'))

if is_simulated:
    st.warning("⚠️ **Network pipe restricted.** Running Safe 2-Year Synthetic Simulation.")
else:
    st.success(f"🟢 **Successfully Synced {len(df)} Real Historical Hours directly via Paginated Binance Pipeline!**")

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

# Dynamic Split Engine (Strict 50:50 Ratio Window on Expanded Data)
split_idx = int(len(df) * 0.50)
df_train = df.iloc[:split_idx]
X_train = df_train[features_matrix].copy()
y_train = df_train['State_Direction'].copy()

df_predict = df.iloc[split_idx:].copy()
X_predict = df_predict[features_matrix].copy()

if len(X_predict) < 5:
    st.error("🚨 Historical stream data array too small for prediction.")
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

    # Feature Importance Math (W%) with Array Bounds Safety
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
    
    if len(feat_weights_arr) > 0 and feat_weights_arr.shape[1] == 5:
        df_predict['W_KalmanDiff(%)_Raw'] = feat_weights_arr[:, 0]
        df_predict['W_OrderImb(%)_Raw'] = feat_weights_arr[:, 1]
        df_predict['W_BodyImb(%)_Raw'] = feat_weights_arr[:, 2]
        df_predict['W_NormGap(%)_Raw'] = feat_weights_arr[:, 3]
        df_predict['W_Velocity(%)_Raw'] = feat_weights_arr[:, 4]
    else:
        for col in ['W_KalmanDiff(%)_Raw', 'W_OrderImb(%)_Raw', 'W_BodyImb(%)_Raw', 'W_NormGap(%)_Raw', 'W_Velocity(%)_Raw']:
            df_predict[col] = 20.0

    # -----------------------------------------------------------------
    # 🧠 DUAL KALMAN TUNNEL & STRICT PRICE CONFIRMATION CROSSOVER LOGIC
    # -----------------------------------------------------------------
    wma_weights = np.arange(12, 0, -1) 
    wma_sum = np.sum(wma_weights)       

    df['Fast_WMA_Tunnel'] = df['b_Kalman_Price'].rolling(window=12).apply(lambda x: np.sum(x * wma_weights) / wma_sum, raw=True)
    df['Slow_WMA_Tunnel'] = df['Slow_Kalman_Price'].rolling(window=12).apply(lambda x: np.sum(x * wma_weights) / wma_sum, raw=True)
    
    df_predict['Slow_Kalman_Price'] = df['Slow_Kalman_Price'].loc[df_predict.index]
    df_predict['Fast_WMA_Tunnel'] = df['Fast_WMA_Tunnel'].loc[df_predict.index]
    df_predict['Slow_WMA_Tunnel'] = df['Slow_WMA_Tunnel'].loc[df_predict.index]

    price_vals = df_predict['a_Close'].to_numpy()
    fast_vals = df_predict['b_Kalman_Price'].to_numpy()
    slow_vals = df_predict['Slow_Kalman_Price'].to_numpy()
    fast_wma = df_predict['Fast_WMA_Tunnel'].to_numpy()
    slow_wma = df_predict['Slow_WMA_Tunnel'].to_numpy()
    
    signal_log = []
    for idx in range(len(fast_vals)):
        if np.isnan(fast_wma[idx]) or np.isnan(slow_wma[idx]):
            signal_log.append("⏳ LOADING")
            continue
            
        # STRICT 4-GATE TRIGGER LOCK 
        fast_bullish = (fast_vals[idx] > fast_wma[idx]) and (price_vals[idx] > fast_vals[idx])
        slow_bullish = (slow_vals[idx] > slow_wma[idx]) and (price_vals[idx] > slow_vals[idx])
        
        fast_bearish = (fast_vals[idx] < fast_wma[idx]) and (price_vals[idx] < fast_vals[idx])
        slow_bearish = (slow_vals[idx] < slow_wma[idx]) and (price_vals[idx] < slow_vals[idx])
        
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

    # Sequential UI Columns Definition Matrix
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

    st.subheader(f"📋 Live 2-Year Aggregated Engine Display Matrix Active")
    st.dataframe(display_df, use_container_width=True, height=750)
