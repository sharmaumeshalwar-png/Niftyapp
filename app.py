import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pytz

# Page Configuration
st.set_page_config(page_title="Nifty IST Slow WMA Prob Engine", layout="wide")
st.title("⚡ Nifty Live 1-Year 1-Hour Standalone Engine [Slow WMA Prob Edition]")
st.write("🎯 **Pure Real-Time Engine:** 1-Year Data Feed | Probabilities Strictly Based on Slow WMA Dynamics")

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
# 🛡️ DIRECT REAL TIME DATA ONLY (1-YEAR TRADING RANGE)
# -----------------------------------------------------------------
df = None
selected_period = "1y"  
selected_interval = "1h" 

with st.spinner("Fetching 1-Year Live Data directly from Exchange Server..."):
    try:
        df = yf.download(tickers="^NSEI", period=selected_period, interval=selected_interval)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 100: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_convert('Asia/Kolkata')
        else:
            st.error("🚨 Error: Insufficient data from exchange server.")
            st.stop()
            
    except Exception as e:
        st.error(f"🚨 API Connection Failed: {e}")
        st.stop()

st.success(f"🟢 **Successfully Synced {len(df)} Real-Time Nifty Spot Candles across 1 Year in IST!**")

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

# TUNNEL CALCULATIONS
wma_weights = np.arange(12, 0, -1) 
wma_sum = np.sum(wma_weights)       
df['Fast_WMA_Tunnel'] = df['b_Kalman_Price'].rolling(window=12).apply(lambda x: np.sum(x * wma_weights) / wma_sum, raw=True)
df['Slow_WMA_Tunnel'] = df['Slow_Kalman_Price'].rolling(window=12).apply(lambda x: np.sum(x * wma_weights) / wma_sum, raw=True)

# VIDYA CALCULATIONS
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

# KALMAN GAP DEV CALCULATION
df['Kalman_Gap_Dev'] = apply_kalman_filter_custom((df['a_Close'] - df['Fast_WMA_Tunnel']).values, initial_p=0.50, q_val=0.001, r_val=0.1)

# Target Direction (1 if next close is higher, 0 if lower)
df['Target_Next_Direction'] = np.where(df['a_Close'].shift(-1) > df['a_Close'], 1, 0)

# Robust forward fill to secure database shapes
df.ffill().bfill()

# Dynamic Split Engine (Strict 1-Year history divided 50:50)
split_idx = int(len(df) * 0.50)
df_train = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

# -----------------------------------------------------------------
# 🤖 SLOW WMA BASED PROBABILITY SOLVER
# -----------------------------------------------------------------
# We define pure Slow WMA behavior features for ML training:
# 1. Close to Slow WMA Tunnel Distance (Gap)
# 2. Slow WMA Tunnel Momentum (Slope/Velocity)
# 3. Position state of Close relative to Slow WMA (Above = 1, Below = 0)
df_train['Slow_WMA_Gap'] = df_train['a_Close'] - df_train['Slow_WMA_Tunnel']
df_train['Slow_WMA_Velocity'] = df_train['Slow_WMA_Tunnel'].diff(1).fillna(0)
df_train['Slow_WMA_State'] = np.where(df_train['a_Close'] > df_train['Slow_WMA_Tunnel'], 1, 0)

df_predict['Slow_WMA_Gap'] = df_predict['a_Close'] - df_predict['Slow_WMA_Tunnel']
df_predict['Slow_WMA_Velocity'] = df_predict['Slow_WMA_Tunnel'].diff(1).fillna(0)
df_predict['Slow_WMA_State'] = np.where(df_predict['a_Close'] > df_predict['Slow_WMA_Tunnel'], 1, 0)

slow_wma_features = ['Slow_WMA_Gap', 'Slow_WMA_Velocity', 'Slow_WMA_State']

# Dynamic Model trained strictly on pure Slow WMA parameters
model_flow = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=1, random_state=42)
model_flow.fit(df_train[slow_wma_features], df_train['Target_Next_Direction'])

# Extracting pure Slow WMA basis probabilities
probabilities = model_flow.predict_proba(df_predict[slow_wma_features])
df_predict['Prob_Down_Raw'] = probabilities[:, 0]
df_predict['Prob_Up_Raw'] = probabilities[:, 1]

# Crossover Logic Engine
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
    fast_bullish = (fast_vals[idx] > fast_wma[idx]) and (price_vals[idx] > fast_vals[idx])
    slow_bullish = (slow_vals[idx] > slow_wma[idx]) and (price_vals[idx] > slow_vals[idx])
    fast_bearish = (fast_vals[idx] < fast_wma[idx]) and (price_vals[idx] < fast_vals[idx])
    slow_bearish = (slow_vals[idx] < slow_wma[idx]) and (price_vals[idx] < slow_vals[idx])
    
    if fast_bullish and slow_bullish: signal_log.append("🟢 BUY")
    elif fast_bearish and slow_bearish: signal_log.append("🔴 SELL")
    else: signal_log.append("⏳ WAIT ZONE")
df_predict['Signal'] = signal_log

# Feature Importance mapping based on pure Slow WMA components
importances = model_flow.feature_importances_
feat_weights = []
X_predict_arr = df_predict[slow_wma_features].to_numpy()
X_train_mean = df_train[slow_wma_features].mean().to_numpy()
X_train_std = df_train[slow_wma_features].std().to_numpy() + 1e-10

for row in X_predict_arr:
    deviation = np.abs(row - X_train_mean) / X_train_std
    raw_contrib = deviation * importances
    total_contrib = np.sum(raw_contrib) + 1e-10
    feat_weights.append((raw_contrib / total_contrib) * 100)

feat_weights_arr = np.array(feat_weights)
if len(feat_weights_arr) > 0 and feat_weights_arr.shape[1] == 3:
    df_predict['W_Gap(%)_Raw'] = feat_weights_arr[:, 0]
    df_predict['W_Velocity(%)_Raw'] = feat_weights_arr[:, 1]
    df_predict['W_State(%)_Raw'] = feat_weights_arr[:, 2]
else:
    for col in ['W_Gap(%)_Raw', 'W_Velocity(%)_Raw', 'W_State(%)_Raw']:
        df_predict[col] = 33.3

# Live Accumulators Tracking Space
prob_up_vals = df_predict['Prob_Up_Raw'].to_numpy()
prob_down_vals = df_predict['Prob_Down_Raw'].to_numpy()
scores_log, raw_weighted_momentum_log = [], []
accumulator = 0
for i in range(len(prob_up_vals)):
    if prob_up_vals[i] >= 0.55: accumulator += 1
    elif prob_down_vals[i] >= 0.55: accumulator -= 1
    accumulator = max(-5, min(5, accumulator))
    scores_log.append(accumulator)
    raw_weighted_momentum_log.append(price_vals[i] - fast_vals[i])

df_predict['Accumulator_Score'] = scores_log  
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum_log, initial_p=0.50, q_val=0.001, r_val=0.1)

# UI Conversion Formatting
df_predict['W_Gap(%)'] = df_predict['W_Gap(%)_Raw'].round(1).astype(str) + "%"
df_predict['W_Velocity(%)'] = df_predict['W_Velocity(%)_Raw'].round(1).astype(str) + "%"
df_predict['W_State(%)'] = df_predict['W_State(%)_Raw'].round(1).astype(str) + "%"

# Display Columns Alignment Matrix (With W_Gap(%), W_Velocity(%) relative to Slow WMA)
clean_display_cols = [
    'a_Close', 'Kalman_Gap_Dev', 'Vidhya', 'Close_Minus_Vidhya', 'VIDYA_Weighted_Momentum', 'VIDYA_Accumulator_Score',
    'b_Kalman_Price', 'Fast_WMA_Tunnel', 'Slow_Kalman_Price', 'Slow_WMA_Tunnel', 'Signal', 
    'Prob_Up_Raw', 'Prob_Down_Raw',
    'W_Gap(%)', 'W_Velocity(%)', 'W_State(%)',
    'Accumulator_Score', 'Weighted_Momentum'
]
display_df = df_predict[clean_display_cols].copy()

for c in ['a_Close', 'Kalman_Gap_Dev', 'Vidhya', 'Close_Minus_Vidhya', 'VIDYA_Weighted_Momentum', 'b_Kalman_Price', 'Fast_WMA_Tunnel', 'Slow_Kalman_Price', 'Slow_WMA_Tunnel', 'Weighted_Momentum']:
    display_df[c] = display_df[c].round(2)
for c in ['Prob_Up_Raw', 'Prob_Down_Raw']:
    display_df[c] = display_df[c].round(3)
    
# Inverting rows for latest candles on top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader(f"📋 Live Nifty Spot Master Matrix [Pure Slow WMA Probability Engine]")
st.dataframe(display_df, use_container_width=True, height=750)
