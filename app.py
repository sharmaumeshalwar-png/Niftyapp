import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier

# Page Configuration
st.set_page_config(page_title="BTC Binance 2-Year Rigid Engine", layout="wide")
st.title("🛡️ BTC Strict 2-Year Multi-Route 200-Point Radar")
st.write("🎯 **Pure 50:50 Learn-Predict Engine:** Smart Fallback Network Activated (No Leakage, Rigid Barfill Swept).")

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
        rs_ratio = r_val / s_val
        h = np.log(rs_ratio) / np.log(window)
        hurst_values[i] = np.clip(h, 0.0, 1.0)
    return hurst_values

# =====================================================================
# 2. HYBRID DATA INGESTION (BINANCE WITH YFINANCE FALLBACK)
# =====================================================================
@st.cache_data(ttl=1800)
def load_2_year_hybrid_data():
    df_out = None
    data_source = "Binance API"
    
    # --- ROUTE A: TRY BINANCE DIRECT FETCH ---
    try:
        url = "https://api.binance.com/api/v3/klines"
        all_candles = []
        end_time = int(time.time() * 1000)
        start_time = end_time - (2 * 365 * 24 * 60 * 60 * 1000)
        current_start = start_time
        
        chunk_count = 0
        while current_start < end_time and chunk_count < 18:
            params = {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1000
            }
            res = requests.get(url, params=params, timeout=10).json()
            if not res or not isinstance(res, list) or len(res) == 0:
                break
            all_candles.extend(res)
            current_start = res[-1][6] + 1
            chunk_count += 1
            time.sleep(0.1)
            
        if len(all_candles) > 100:
            data_matrix = []
            for candle in all_candles:
                data_matrix.append([
                    pd.to_datetime(candle[0], unit='ms'),
                    float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4])
                ])
            df_out = pd.DataFrame(data_matrix, columns=['Time', 'Open', 'High', 'Low', 'Close'])
            df_out.set_index('Time', inplace=True)
    except Exception:
        df_out = None

    # --- ROUTE B: FALLBACK TO YFINANCE IF ROUTE A BLOCKED ---
    if df_out is None or len(df_out) < 100:
        data_source = "Yahoo Finance (Rigid Locked Routing)"
        try:
            raw_yf = yf.download(tickers="BTC-USD", period="2y", interval="1h")
            if not raw_yf.empty:
                df_out = pd.DataFrame(index=raw_yf.index)
                df_out['Open'] = raw_yf['Open'].values.astype(float)
                df_out['High'] = raw_yf['High'].values.astype(float)
                df_out['Low'] = raw_yf['Low'].values.astype(float)
                df_out['Close'] = raw_yf['Close'].values.astype(float)
        except Exception as e:
            st.error(f"🚨 Both Ingestion Nodes Failed: {e}")
            st.stop()
            
    if df_out is None or len(df_out) == 0:
        st.error("🚨 Ingestion Pipeline Failure: No market data parsed.")
        st.stop()
        
    # --- RIGID CHECK: BARFILL & INTEGRITY SWEEP ---
    df_out = df_out.ffill()
    df_out.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    
    # Handle Timezone conversions cleanly
    if df_out.index.tz is None:
        df_out.index = df_out.index.tz_localize('UTC')
    df_out.index = df_out.index.tz_convert('Asia/Kolkata')
    
    return df_out, data_source

df, active_source = load_2_year_hybrid_data()

# =====================================================================
# 3. STRICT 200-POINT ABSOLUTE RANGE LOCKING ENGINE
# =====================================================================
raw_closes = df['Close'].to_numpy(dtype=float)
raw_times = df.index
range_size = 200.0

range_closes = [raw_closes[0]]
range_times = [raw_times[0]]
range_directions = ["INITIAL"]

current_anchor = raw_closes[0]

for i in range(1, len(raw_closes)):
    price_diff = raw_closes[i] - current_anchor
    
    if abs(price_diff) >= range_size:
        num_bars = int(abs(price_diff) // range_size)
        direction = np.sign(price_diff)
        
        for _ in range(num_bars):
            current_anchor += direction * range_size
            range_closes.append(current_anchor)
            range_times.append(raw_times[i])
            range_directions.append("UP" if direction > 0 else "DOWN")

df_range = pd.DataFrame(index=range_times, data={
    'Close': range_closes,
    'Direction_State': range_directions
})

# =====================================================================
# 4. FIXED FEATURE EXTRACTION & SCALING
# =====================================================================
close_arr = df_range['Close'].to_numpy(dtype=float)
df_range['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
df_range['Hurst'] = calculate_rolling_hurst(close_arr, window=50)

raw_weighted_momentum = df_range['Close'] - df_range['Kalman_Baseline']
df_range['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.to_numpy(), initial_p=0.50, q_val=0.001, r_val=0.1)

df_range['Hurst_Amp_Momentum'] = df_range['Weighted_Momentum'] * (df_range['Hurst'] * 2.0)
df_range['Normalized_Dev_Scaled'] = ((df_range['Close'] - df_range['Kalman_Baseline']) / df_range['Kalman_Baseline']) * 1000.0

df_range.dropna(subset=['Hurst', 'Close', 'Normalized_Dev_Scaled'], inplace=True)

# =====================================================================
# 5. ML TARGET LABEL GENERATION (ZERO-LEAKAGE PROTECTION)
# =====================================================================
n_len = len(df_range)
target_labels = []
dir_states = df_range['Direction_State'].to_numpy()

for idx in range(n_len):
    if idx < n_len - 1:
        next_state = dir_states[idx + 1]
        target_labels.append(1 if next_state == "UP" else (2 if next_state == "DOWN" else 0))
    else:
        target_labels.append(0)

df_range['Target_Class'] = target_labels

# =====================================================================
# 6. PURE 50:50 LEARN VS PREDICT SPLIT
# =====================================================================
mid_point = n_len // 2
train_df = df_range.iloc[:mid_point - 1].dropna()
predict_df = df_range.iloc[mid_point:].copy()

if len(train_df) > 10 and len(predict_df) > 0:
    X_train = train_df[['Hurst_Amp_Momentum', 'Normalized_Dev_Scaled']].to_numpy()
    y_train = train_df['Target_Class'].to_numpy()
    
    tree_model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=4, class_weight='balanced', random_state=42)
    tree_model.fit(X_train, y_train)
    
    X_predict = predict_df[['Hurst_Amp_Momentum', 'Normalized_Dev_Scaled']].to_numpy()
    predictions_out = tree_model.predict(X_predict)
else:
    predictions_out = np.zeros(len(predict_df))

state_map = {0: "⏳ WAITING FOR BREAKOUT", 1: "📈 NEXT BAR: LOCKED UP", 2: "📉 NEXT BAR: LOCKED DOWN"}

final_display_matrix = []
predict_len = len(predict_df)

for i in range(predict_len):
    base_state = state_map[predictions_out[i]]
    if i == predict_len - 1:
        final_display_matrix.append(f"🟢 LIVE PROJECTION -> {base_state}")
    else:
        final_display_matrix.append(f"📊 HISTORICAL OOS -> {base_state}")

predict_df['🌲 ML Tree Decision Grid'] = final_display_matrix

# =====================================================================
# 7. LIVE USER INTERFACE
# =====================================================================
latest_row = predict_df.iloc[-1]

st.markdown("---")
st.subheader("🌲 MACHINE LEARNING RIGID MATRIX OUTPUT BOARD")

if "LOCKED UP" in latest_row['🌲 ML Tree Decision Grid']:
    st.success(f"### {latest_row['🌲 ML Tree Decision Grid']}")
elif "LOCKED DOWN" in latest_row['🌲 ML Tree Decision Grid']:
    st.info(f"### {latest_row['🌲 ML Tree Decision Grid']}")
else:
    st.warning(f"### {latest_row['🌲 ML Tree Decision Grid']}")

st.markdown("---")
st.sidebar.markdown(f"### 🛡️ Core Network Routing")
st.sidebar.info(f"Connected Via: {active_source}")
st.sidebar.success("✓ Dynamic Crash Guard Active")
st.sidebar.success("✓ Zero-Leakage Active")

st.sidebar.metric(label="📊 Total 2-Yr Base Candles", value=f"{len(df)}")
st.sidebar.metric(label="📈 Total Generated Locked Bars", value=f"{n_len}")
st.sidebar.metric(label="🧠 50% Learn Model Rows", value=f"{len(train_df)}")

r_col1, r_col2 = st.columns(2)
with r_col1:
    st.metric(label="⚙️ Real-Time Dynamic HAM Log", value=f"{latest_row['Hurst_Amp_Momentum']:.4f}")
with r_col2:
    st.metric(label="📊 Live System Anchor Price", value=f"${latest_row['Close']:.2f}")

clean_cols = ['Close', 'Direction_State', 'Hurst_Amp_Momentum', '🌲 ML Tree Decision Grid']
display_df = predict_df[clean_cols].copy()
display_df.rename(columns={
    'Close': 'Locked Anchor Price', 
    'Direction_State': 'Current Bar State',
    'Hurst_Amp_Momentum': 'Dynamic HAM Log',
    '🌲 ML Tree Decision Grid': 'ML Signal Grid'
}, inplace=True)

display_df_inverted = display_df.iloc[::-1].copy()
display_df_inverted.index = display_df_inverted.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Pure Immutable Prediction Logs")
st.dataframe(display_df_inverted, use_container_width=True, height=500)
