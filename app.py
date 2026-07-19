import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier

# Page Configuration
st.set_page_config(page_title="BTC Raw 1HR Anti-Leakage Engine", layout="wide")
st.title("🛡️ BTC Strict Anti-Leakage 1-Hour Raw Radar")
st.write("🎯 **Pure 50:50 Train/Predict Split:** 1-Hour Normal Candles (Zero Feature Leakage).")

# =====================================================================
# 1. MATHEMATICAL ENGINES
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0: 
        return np.array([])
    x, p = data_array[0], initial_p  
    filtered_values = np.zeros(len(data_array))
    for i, z in enumerate(data_array):
        p = p + q_val
        k = p / (p + r_val)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values[i] = x
    return filtered_values

def calculate_rolling_hurst(price_series, window=50):
    hurst_values = np.full(len(price_series), 0.5) 
    if len(price_series) <= window:
        return hurst_values
    
    log_returns = np.log(price_series / np.roll(price_series, 1))
    log_returns[0] = 0
    
    for i in range(window, len(price_series)):
        window_data = log_returns[i-window+1:i+1]
        cum_dev = np.cumsum(window_data - np.mean(window_data))
        r_val = np.max(cum_dev) - np.min(cum_dev)
        s_val = np.std(window_data) + 1e-10
        
        rs_ratio = max(r_val / s_val, 1e-12)
        h = np.log(rs_ratio) / np.log(window)
        hurst_values[i] = np.clip(h, 0.0, 1.0)
    return hurst_values

# =====================================================================
# 2. DATA INGESTION & BARFILL INTEGRITY CHECK
# =====================================================================
df = None
with st.spinner("Ingesting 2-Year Hourly Market History..."):
    try:
        df = yf.download(tickers="BTC-USD", period="2y", interval="1h", multi_level_index=False)
        if df is None or df.empty:
            st.error("🚨 Error: Data download failed from Yahoo Finance.")
            st.stop()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ingestion Verification & Strict Barfill
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], how='all', inplace=True)
        df = df.ffill()  
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            df.index = df.index.tz_convert('Asia/Kolkata')
            
    except Exception as e:
        st.error(f"🚨 Ingestion Failure: {e}")
        st.stop()

# =====================================================================
# 3. IMMUTABLE FEATURE EXTRACTION & SCALING (RAW 1-HR CANDLES)
# =====================================================================
close_arr = df['Close'].to_numpy(dtype=float)

# Core Baseline & Hurst Calculators
df['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
df['Hurst'] = calculate_rolling_hurst(close_arr, window=50)

# 1-Hour Base Features (Zero Double-Smoothing Lag)
df['Weighted_Momentum'] = df['Close'] - df['Kalman_Baseline']
df['Hurst_Amp_Momentum'] = df['Weighted_Momentum'] * (df['Hurst'] * 2.0)
df['Normalized_Dev_Scaled'] = ((df['Close'] - df['Kalman_Baseline']) / df['Kalman_Baseline']) * 1000.0

# Safe trim indicator warm-up phase
df.dropna(subset=['Hurst', 'Kalman_Baseline', 'Hurst_Amp_Momentum'], inplace=True)

# =====================================================================
# 4. ML TARGET LABEL GENERATION (NEXT-BAR NORMAL DIRECTION)
# =====================================================================
n_len = len(df)
target_labels = []
open_arr = df['Open'].to_numpy()
close_arr = df['Close'].to_numpy()

for idx in range(n_len):
    if idx < n_len - 1:
        next_close = close_arr[idx + 1]
        next_open = open_arr[idx + 1]
        
        # Target based on strict raw candle direction
        if next_close > next_open:
            target_labels.append(1)  # Bullish Raw Candle
        else:
            target_labels.append(2)  # Bearish Raw Candle
    else:
        target_labels.append(0) 

df['Target_Class'] = target_labels

# =====================================================================
# 5. PURE 50:50 LEARN VS PREDICT ENGINE (STRICT FRONT SHIELD)
# =====================================================================
mid_point = n_len // 2

train_df = df.iloc[:mid_point - 1].dropna()
predict_df = df.iloc[mid_point:].copy()

# Integrity Check Boundary Protection
overlap_check = train_df.index.intersection(predict_df.index)
if len(overlap_check) > 0 and train_df.index.max() >= predict_df.index.min():
    st.error("🚨 CRITICAL LEAKAGE DETECTED: Timeline cross-over observed!")
    st.stop()

if len(train_df) > 10 and len(predict_df) > 0:
    X_train = train_df[['Hurst_Amp_Momentum', 'Normalized_Dev_Scaled']].to_numpy()
    y_train = train_df['Target_Class'].to_numpy()
    
    tree_model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight='balanced', random_state=42)
    tree_model.fit(X_train, y_train)
    
    X_predict = predict_df[['Hurst_Amp_Momentum', 'Normalized_Dev_Scaled']].to_numpy()
    predictions_out = tree_model.predict(X_predict)
else:
    predictions_out = np.zeros(len(predict_df))

state_map = {0: "⏳ SCANNING MATRIX", 1: "📈 CANDLE: GREEN (UP)", 2: "📉 CANDLE: RED (DOWN)"}

final_display_matrix = []
predict_len = len(predict_df)

for i in range(predict_len):
    base_state = state_map[predictions_out[i]]
    if i == predict_len - 1:
        final_display_matrix.append(f"🟢 LIVE PROJECTION -> {base_state}")
    else:
        final_display_matrix.append(f"📊 HISTORICAL OOS -> {base_state}")

predict_df['ML Signal Grid'] = final_display_matrix

# =====================================================================
# 6. STAGE VIEW RENDER (STRICT COLUMN FOCUS)
# =====================================================================
latest_row = predict_df.iloc[-1]

st.markdown("---")
st.subheader("🌲 MACHINE LEARNING RIGID MATRIX OUTPUT (1-HOUR RAW MODE)")

if "GREEN" in latest_row['ML Signal Grid']:
    st.success(f"### {latest_row['ML Signal Grid']}")
elif "RED" in latest_row['ML Signal Grid']:
    st.info(f"### {latest_row['ML Signal Grid']}")
else:
    st.warning(f"### {latest_row['ML Signal Grid']}")

st.markdown("---")
st.sidebar.markdown(f"### 🛡️ Pipeline Integrity Audit")
st.sidebar.success("✓ 1hr Raw Stream Bound")
st.sidebar.success("✓ Pure Candles (No HA Smoothing)")
st.sidebar.success("✓ 50:50 Temporal Shield Active")

st.sidebar.markdown(f"### 📊 Data Split Analytics")
st.sidebar.metric(label="📊 Total 1HR Candlesticks", value=f"{n_len}")
st.sidebar.metric(label="🧠 50% Training Samples", value=f"{len(train_df)}")
st.sidebar.metric(label="👁️ 50% Validation Frontier", value=f"{predict_len}")

r_col1, r_col2, r_col3 = st.columns(3)
with r_col1:
    st.metric(label="📊 Live Close Price", value=f"${latest_row['Close']:.2f}")
with r_col2:
    st.metric(label="⚡ Dynamic Weighted Momentum", value=f"{latest_row['Weighted_Momentum']:.4f}")
with r_col3:
    st.metric(label="⚙️ Hurst Amplitude Log (H.A.M)", value=f"{latest_row['Hurst_Amp_Momentum']:.4f}")

# Strict Table Columns Filtering (Date, Close, Weighted Momentum, Hurst, H.A.M, Matrix Signal)
clean_cols = ['Close', 'Weighted_Momentum', 'Hurst', 'Hurst_Amp_Momentum', 'ML Signal Grid']
display_df = predict_df[clean_cols].copy()

display_df.rename(columns={
    'Close': 'Close Price', 
    'Weighted_Momentum': 'Weighted Momentum',
    'Hurst': 'Hurst Exponent',
    'Hurst_Amp_Momentum': 'H.A.M Log',
    'ML Signal Grid': 'Signal Direction Target'
}, inplace=True)

# Inverting matrix display log for top live reading
display_df_inverted = display_df.iloc[::-1].copy()
display_df_inverted.index = display_df_inverted.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Out-of-Sample Prediction Logs (Strict Raw Columns)")
st.dataframe(display_df_inverted, use_container_width=True, height=500)
