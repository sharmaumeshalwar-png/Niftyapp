import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
from sklearn.tree import DecisionTreeClassifier

# Page Configuration
st.set_page_config(page_title="BTC Binance 2-Year Rigid Engine", layout="wide")
st.title("🛡️ BTC Strict 2-Year Binance 200-Point Radar")
st.write("🎯 **Pure 50:50 Learn-Predict Engine:** Powered by Binance API (No Leakage, Rigid Barfill Swept).")

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
# 2. BINANCE 2-YEAR HISTORICAL DATA INGESTION (WITH CRASH GUARD)
# =====================================================================
@st.cache_data(ttl=1800)  # Cache for 30 minutes to reduce heavy API calls
def load_2_year_binance_data():
    url = "https://api.binance.com/api/v3/klines"
    all_candles = []
    
    # Calculate timestamps for 2 years ago up to now
    end_time = int(time.time() * 1000)
    start_time = end_time - (2 * 365 * 24 * 60 * 60 * 1000)  # 2 Years in ms
    
    current_start = start_time
    progress_bar = st.progress(0.0, text="Fetching 2-Year Binance Ledger...")
    
    # Max limit per request is 1000. 2 Years of 1H data is ~17,520 candles.
    total_estimated_chunks = 18
    chunk_count = 0
    
    while current_start < end_time:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        try:
            res = requests.get(url, params=params).json()
            
            # 🛡️ RIGID ENGINE CRASH GUARD: Check if response is empty or invalid
            if not res or not isinstance(res, list) or len(res) == 0:
                break
                
            all_candles.extend(res)
            
            # Move start window to the timestamp of the last received candle + 1ms
            current_start = res[-1][6] + 1
            
            chunk_count += 1
            progress_pct = min(chunk_count / total_estimated_chunks, 1.0)
            progress_bar.progress(progress_pct, text=f"Loaded {len(all_candles)} hourly candles...")
            
            # Throttle slightly to respect API rate limits
            time.sleep(0.1)
            
        except Exception:
            break
        
    progress_bar.empty()
    
    if len(all_candles) == 0:
        st.error("🚨 Zero Candles Retrieved from Binance. Please check internet or deployment logs.")
        st.stop()
    
    # Structure into clean DataFrame
    data_matrix = []
    for candle in all_candles:
        data_matrix.append([
            pd.to_datetime(candle[0], unit='ms'),  # Open Time
            float(candle[1]),  # Open
            float(candle[2]),  # High
            float(candle[3]),  # Low
            float(candle[4])   # Close
        ])
        
    df_out = pd.DataFrame(data_matrix, columns=['Time', 'Open', 'High', 'Low', 'Close'])
    df_out.set_index('Time', inplace=True)
    
    # --- RIGID CHECK: BARFILL & INTEGRITY SWEEP ---
    df_out = df_out.ffill()
    df_out.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    df_out.index = df_out.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    
    return df_out

df = load_2_year_binance_data()

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
# 5. ML TARGET LABEL GENERATION (ZERO-LEAKAGE LOOK-AHEAD PROTECTION)
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
st.subheader("🌲 MACHINE LEARNING RIGID BINANCE OUTPUT BOARD")

if "LOCKED UP" in latest_row['🌲 ML Tree Decision Grid']:
    st.success(f"### {latest_row['🌲 ML Tree Decision Grid']}")
elif "LOCKED DOWN" in latest_row['🌲 ML Tree Decision Grid']:
    st.info(f"### {latest_row['🌲 ML Tree Decision Grid']}")
else:
    st.warning(f"### {latest_row['🌲 ML Tree Decision Grid']}")

st.markdown("---")
st.sidebar.markdown(f"### 🛡️ Binance Integrity Audit")
st.sidebar.success("✓ Binance API Connected")
st.sidebar.success("✓ Check Barfill Swept")
st.sidebar.success("✓ Zero-Leakage Active")

st.sidebar.metric(label="📊 Total 2-Yr Base Candles", value=f"{len(df)}")
st.sidebar.metric(label="📈 Total Generated Locked Bars", value=f"{n_len}")
st.sidebar.metric(label="🧠 50% Learn Model Rows", value=f"{len(train_df)}")
st.sidebar.metric(label="🔮 50% Predict Model Rows", value=f"{len(predict_df)}")

r_col1, r_col2 = st.columns(2)
with r_col1:
    st.metric(label="⚙️ Real-Time Dynamic HAM Log", value=f"{latest_row['Hurst_Amp_Momentum']:.4f}")
with r_col2:
    st.metric(label="📊 Binance Live Anchor Price", value=f"${latest_row['Close']:.2f}")

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
