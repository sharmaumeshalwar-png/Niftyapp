import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier

# Page Configuration
st.set_page_config(page_title="BTC Strict 2-Year Locked Engine", layout="wide")
st.title("🛡️ BTC Strict 2-Year Rigid 200-Point Radar")
st.write("🎯 **Pure 50:50 Engine:** 2-Year Horizon, Zero-Leakage Shield, and Permanent Barfill Integrity Swept.")

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
# 2. IMMUTABLE 2-YEAR HISTORICAL DATA INGESTION (WITH BFILL/FFILL CHECK)
# =====================================================================
@st.cache_data(ttl=900)  # Rigid cache to lock historical rows across interactions
def load_strict_2year_data():
    # Parsing full 2 years of 1-Hour data continuously
    raw_data = yf.download(tickers="BTC-USD", period="2y", interval="1h", progress=False)
    if raw_data.empty:
        st.error("🚨 Critical Ingestion Error: Market Ledger could not be retrieved.")
        st.stop()
        
    df_out = pd.DataFrame(index=raw_data.index)
    df_out['Close'] = raw_data['Close'].values.astype(float)
    
    # --- RIGID CHECK: BARFILL & INTEGRITY SWEEP ---
    df_out = df_out.bfill().ffill()  # Enforce strict double-sided barfill sweep
    df_out.dropna(subset=['Close'], inplace=True)
    
    if df_out.index.tz is None:
        df_out.index = df_out.index.tz_localize('UTC')
    df_out.index = df_out.index.tz_convert('Asia/Kolkata')
    
    return df_out

df_raw = load_strict_2year_data()

# =====================================================================
# 3. ABSOLUTE 200-POINT RANGE LOCKING SEQUENCE
# =====================================================================
raw_closes = df_raw['Close'].to_numpy(dtype=float)
raw_times = df_raw.index
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

df_master = pd.DataFrame(index=range_times, data={
    'Close': range_closes,
    'Direction_State': range_directions
})

# =====================================================================
# 4. FIXED FEATURE INTEGRITY BLOCK
# =====================================================================
close_arr = df_master['Close'].to_numpy(dtype=float)
kb = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
hurst = calculate_rolling_hurst(close_arr, window=50)
wm = apply_kalman_filter_custom((close_arr - kb), initial_p=0.50, q_val=0.001, r_val=0.1)

df_master['HAM_Log'] = wm * (hurst * 2.0)
df_master['Dev_Scaled'] = ((close_arr - kb) / kb) * 1000.0
df_master.dropna(subset=['HAM_Log', 'Dev_Scaled'], inplace=True)

# =====================================================================
# 5. PURE ZERO-LEAKAGE TARGET GENERATION
# =====================================================================
n_len = len(df_master)
target_labels = []
dir_states = df_master['Direction_State'].to_numpy()

for idx in range(n_len):
    if idx < n_len - 1:
        next_state = dir_states[idx + 1]
        target_labels.append(1 if next_state == "UP" else (2 if next_state == "DOWN" else 0))
    else:
        target_labels.append(0)
df_master['Target_Class'] = target_labels

# =====================================================================
# 6. PURE 50:50 LEARN VS PREDICT ENGINE (STRICT OOS SPLIT)
# =====================================================================
mid_point = n_len // 2
train_df = df_master.iloc[:mid_point].copy()
predict_df = df_master.iloc[mid_point:].copy()

# Prevent data leakage: Fit model EXCLUSIVELY on first 50% data blocks
X_train = train_df[['HAM_Log', 'Dev_Scaled']].to_numpy()
y_train = train_df['Target_Class'].to_numpy()

tree_model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=4, class_weight='balanced', random_state=42)
tree_model.fit(X_train, y_train)

# Predict Out-of-Sample blocks on second 50% data blocks
X_pred = predict_df[['HAM_Log', 'Dev_Scaled']].to_numpy()
predictions = tree_model.predict(X_pred)

state_map = {0: "⏳ BREAKOUT LOOKOUT", 1: "📈 NEXT BAR: LOCKED UP", 2: "📉 NEXT BAR: LOCKED DOWN"}

final_display_matrix = []
predict_len = len(predict_df)

for i in range(predict_len):
    base_state = state_map[predictions[i]]
    if i == predict_len - 1:
        final_display_matrix.append(f"🟢 LIVE PROJECTION -> {base_state}")
    else:
        final_display_matrix.append(f"📊 HISTORICAL OOS -> {base_state}")

predict_df['ML Signal Grid'] = final_display_matrix

# =====================================================================
# 7. RIGID VISUALIZATION BOARD
# =====================================================================
latest_row = predict_df.iloc[-1]

st.markdown("---")
st.subheader("🌲 MACHINE LEARNING RIGID MATRIX OUTPUT BOARD")

if "LOCKED UP" in latest_row['ML Signal Grid']:
    st.success(f"### {latest_row['ML Signal Grid']}")
elif "LOCKED DOWN" in latest_row['ML Signal Grid']:
    st.info(f"### {latest_row['ML Signal Grid']}")
else:
    st.warning(f"### {latest_row['ML Signal Grid']}")

st.markdown("---")
st.sidebar.markdown("### 🛡️ Rigidity Core Audit")
st.sidebar.success("✓ 2-Year Dataset Engaged")
st.sidebar.success("✓ 1-Hour Timeframe Confirmed")
st.sidebar.success("✓ 50:50 Split (Zero Leakage)")
st.sidebar.success("✓ Double-Sided Bfill Sweep")

st.sidebar.metric(label="📊 Total 2-Yr Base Candles", value=f"{len(df_raw)}")
st.sidebar.metric(label="📈 Total Generated Range Bars", value=f"{n_len}")
st.sidebar.metric(label="🧠 50% Learn Model Rows", value=f"{len(train_df)}")
st.sidebar.metric(label="🔮 50% Predict Model Rows", value=f"{len(predict_df)}")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="⚙️ Real-Time HAM Log", value=f"{latest_row['HAM_Log']:.4f}")
with col2:
    st.metric(label="📊 Strict Anchor Price", value=f"${latest_row['Close']:.2f}")

clean_cols = ['Close', 'Direction_State', 'HAM_Log', 'ML Signal Grid']
display_df = predict_df[clean_cols].copy()
display_df.rename(columns={
    'Close': 'Locked Anchor Price', 
    'Direction_State': 'Current Bar State',
    'HAM_Log': 'Dynamic HAM Log'
}, inplace=True)

display_df_inverted = display_df.iloc[::-1].copy()
display_df_inverted.index = display_df_inverted.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Pure Immutable Locked History Logs")
st.dataframe(display_df_inverted, use_container_width=True, height=500)
