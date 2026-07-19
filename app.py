import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
from sklearn.tree import DecisionTreeClassifier

# Page Configuration
st.set_page_config(page_title="BTC Strict State-Lock Engine", layout="wide")
st.title("🛡️ BTC Absolute State-Lock Immutable Radar")
st.write("🎯 **Pure Append-Only Matrix:** Features are calculated once and frozen forever. No Repainting.")

# =====================================================================
# 1. FIXED MATHEMATICAL ENGINES
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
# 2. STATE-LOCK STORAGE INITIALIZATION (THE IMMUTABLE DATABASE)
# =====================================================================
if 'frozen_matrix' not in st.session_state:
    st.session_state.frozen_matrix = pd.DataFrame(columns=[
        'Locked Anchor Price', 'Current Bar State', 'Dynamic HAM Log', 
        'Normalized_Dev_Scaled', 'ML Signal Grid'
    ])
if 'last_processed_time' not in st.session_state:
    st.session_state.last_processed_time = None

# =====================================================================
# 3. BINANCE LEDGER FETCH (RESTRICTED DIRECT INGESTION)
# =====================================================================
url = "https://api.binance.com/api/v3/klines"
params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 600}

try:
    res = requests.get(url, params=params, timeout=10).json()
    if not res or not isinstance(res, list) or len(res) == 0:
        st.error("🚨 Binance Ledger connection dropped. Refreshing...")
        st.stop()
        
    data_matrix = []
    for candle in res:
        data_matrix.append([
            pd.to_datetime(candle[0], unit='ms'),
            float(candle[4])  # Close Price
        ])
    df_raw = pd.DataFrame(data_matrix, columns=['Time', 'Close']).set_index('Time')
    df_raw.index = df_raw.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
except Exception as e:
    st.error(f"🚨 Network Node Blocked: {e}")
    st.stop()

# =====================================================================
# 4. RIGID BREAKOUT & IMMUTABLE FEATURE BLOCKING
# =====================================================================
raw_closes = df_raw['Close'].to_numpy(dtype=float)
raw_times = df_raw.index
range_size = 200.0

range_closes = [raw_closes[0]]
range_times = [raw_times[0]]
range_directions = ["INITIAL"]
current_anchor = raw_closes[0]

# Generate base ranges sequentially
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

# Process Master Data Frame
df_master = pd.DataFrame(index=range_times, data={
    'Close': range_closes,
    'Direction_State': range_directions
})

# Calculate Engine Analytics globally for historic tracking
close_arr = df_master['Close'].to_numpy(dtype=float)
kb = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
hurst = calculate_rolling_hurst(close_arr, window=50)
wm = apply_kalman_filter_custom((close_arr - kb), initial_p=0.50, q_val=0.001, r_val=0.1)

df_master['HAM_Log'] = wm * (hurst * 2.0)
df_master['Dev_Scaled'] = ((close_arr - kb) / kb) * 1000.0
df_master.dropna(subset=['HAM_Log', 'Dev_Scaled'], inplace=True)

# Train Static Core ML Model on historical blocks to generate clean signals
n_len = len(df_master)
mid_point = n_len // 2

train_df = df_master.iloc[:mid_point].copy()
predict_df = df_master.iloc[mid_point:].copy()

target_labels = []
dir_states = train_df['Direction_State'].to_numpy()
for idx in range(len(train_df)):
    if idx < len(train_df) - 1:
        next_state = dir_states[idx + 1]
        target_labels.append(1 if next_state == "UP" else (2 if next_state == "DOWN" else 0))
    else:
        target_labels.append(0)
train_df['Target'] = target_labels

X_train = train_df[['HAM_Log', 'Dev_Scaled']].to_numpy()
y_train = train_df['Target'].to_numpy()
tree_model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=4, class_weight='balanced', random_state=42)
tree_model.fit(X_train, y_train)

# Predict Out-of-Sample blocks
X_pred = predict_df[['HAM_Log', 'Dev_Scaled']].to_numpy()
predictions = tree_model.predict(X_pred)

state_map = {0: "⏳ BREAKOUT LOOKOUT", 1: "📈 NEXT BAR: LOCKED UP", 2: "📉 NEXT BAR: LOCKED DOWN"}

# =====================================================================
# 5. THE APPEND-ONLY PERSISTENCE GATEWAY (FREEZE LOGIC)
# =====================================================================
fresh_records = []
for idx in range(len(predict_df)):
    t_stamp = predict_df.index[idx].strftime('%Y-%m-%d %H:%M')
    is_live = (idx == len(predict_df) - 1)
    
    sig_label = f"🟢 LIVE PROJECTION -> {state_map[predictions[idx]]}" if is_live else f"📊 HISTORICAL OOS -> {state_map[predictions[idx]]}"
    
    record = {
        'Time': t_stamp,
        'Locked Anchor Price': round(predict_df['Close'].iloc[idx], 2),
        'Current Bar State': predict_df['Direction_State'].iloc[idx],
        'Dynamic HAM Log': round(predict_df['HAM_Log'].iloc[idx], 4),
        'Normalized_Dev_Scaled': round(predict_df['Dev_Scaled'].iloc[idx], 4),
        'ML Signal Grid': sig_label
    }
    fresh_records.append(record)

df_fresh = pd.DataFrame(fresh_records).set_index('Time')

# Build memory: Keep old calculations intact, update or append new blocks only
if st.session_state.frozen_matrix.empty:
    st.session_state.frozen_matrix = df_fresh
else:
    # Retain all historical closed rows exactly as they were captured in previous sessions
    historical_static = st.session_state.frozen_matrix.iloc[:-1]
    
    # Identify new rows that have entered the database pipeline
    new_incoming = df_fresh.loc[~df_fresh.index.isin(historical_static.index)]
    
    # Merge and commit to local state storage
    st.session_state.frozen_matrix = pd.concat([historical_static, new_incoming])

# Ensure full tracking display order
display_final = st.session_state.frozen_matrix.iloc[::-1]

# =====================================================================
# 6. APP INTERFACE DISPLAY BOARD
# =====================================================================
latest_row = df_fresh.iloc[-1]

st.markdown("---")
st.subheader("🌲 IMMUTABLE STATE-LOCK RUNNING BOARD")

if "LOCKED UP" in latest_row['ML Signal Grid']:
    st.success(f"### {latest_row['ML Signal Grid']}")
elif "LOCKED DOWN" in latest_row['ML Signal Grid']:
    st.info(f"### {latest_row['ML Signal Grid']}")
else:
    st.warning(f"### {latest_row['ML Signal Grid']}")

st.markdown("---")
st.sidebar.markdown("### 🛡️ State Memory Audit")
st.sidebar.success("✓ Database Append-Only Active")
st.sidebar.success("✓ Yahoo Route Bypassed")
st.sidebar.metric(label="📊 Frozen Rows Count", value=f"{len(st.session_state.frozen_matrix)}")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="⚙️ Frozen HAM Log (Latest)", value=f"{latest_row['Dynamic HAM Log']:.4f}")
with col2:
    st.metric(label="📊 Frozen System Anchor Price", value=f"${latest_row['Locked Anchor Price']:.2f}")

st.subheader("📋 Pure Immutable Locked History Logs")
st.dataframe(display_final[['Locked Anchor Price', 'Current Bar State', 'Dynamic HAM Log', 'ML Signal Grid']], use_container_width=True, height=500)
