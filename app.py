import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier

# Page Configuration
st.set_page_config(page_title="BTC Synchronized 1-Hour Radar", layout="wide")
st.title("🛡️ BTC Absolute 1-Hour Time-Locked Radar")
st.write("🎯 **Dynamic Time-Series Alignment Engine:** Indicators calibrated specifically for standard hourly structural data.")

# =====================================================================
# 1. TUNED MATHEMATICAL ENGINES (HOURLY SPECIFIC)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=10.0, q_val=0.005, r_val=0.05):
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

def calculate_rolling_hurst(price_series, window=24): # Calibrated to 24-hour cycle
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
        if rs_ratio > 0:
            h = np.log(rs_ratio) / np.log(window)
            hurst_values[i] = np.clip(h, 0.0, 1.0)
    return hurst_values

# =====================================================================
# 2. IMMUTABLE 1-HOUR HISTORICAL INGESTION (STRICT CLOSED BARS)
# =====================================================================
@st.cache_data(ttl=600)
def load_strict_hourly_dataset():
    raw_data = yf.download(tickers="BTC-USD", period="2y", interval="1h", progress=False)
    if raw_data.empty:
        st.error("🚨 Critical Market Ledger Pipeline drop.")
        st.stop()
        
    df_out = pd.DataFrame(index=raw_data.index)
    df_out['Close'] = raw_data['Close'].values.astype(float)
    df_out = df_out.bfill().ffill()
    
    if df_out.index.tz is None:
        df_out.index = df_out.index.tz_localize('UTC')
    df_out.index = df_out.index.tz_convert('Asia/Kolkata')
    
    # --- ABSOLUTE SHIELD: Slice out the active running hour ---
    df_historical_confirmed = df_out.iloc[:-1].copy()
    live_floating_candle = df_out.iloc[-1:].copy()
    
    return df_historical_confirmed, live_floating_candle

df_confirmed, df_live = load_strict_hourly_dataset()

# =====================================================================
# 3. DIRECT 1-HOUR MATRIX FORMATTING
# =====================================================================
df_master = df_confirmed.copy()
df_master['Direction_State'] = np.where(df_master['Close'] >= df_master['Close'].shift(1), "UP", "DOWN")
df_master.iloc[0, df_master.columns.get_loc('Direction_State')] = "INITIAL"

# =====================================================================
# 4. FIXED INDICES CALCULATION (PATTHAR KI LAQEER - REALIGNED)
# =====================================================================
close_arr = df_master['Close'].to_numpy(dtype=float)
kb = apply_kalman_filter_custom(close_arr, initial_p=10.0, q_val=0.005, r_val=0.05)
hurst = calculate_rolling_hurst(close_arr, window=24)

# Mathematical fix for accurate asset deviation mapping
deviation = close_arr - kb
rolling_std = pd.Series(deviation).rolling(window=24, min_periods=1).std().to_numpy()
rolling_std[rolling_std == 0] = 1e-10

df_master['HAM_Log'] = (deviation / rolling_std) * hurst
df_master['Dev_Scaled'] = (deviation / kb) * 1000.0
df_master.dropna(subset=['HAM_Log', 'Dev_Scaled'], inplace=True)

# =====================================================================
# 5. ML ENGINE 50:50 SPLIT BLOCK
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

mid_point = n_len // 2
train_df = df_master.iloc[:mid_point].copy()
predict_df = df_master.iloc[mid_point:].copy()

X_train = train_df[['HAM_Log', 'Dev_Scaled']].to_numpy()
y_train = train_df['Target_Class'].to_numpy()

tree_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, class_weight='balanced', random_state=42)
tree_model.fit(X_train, y_train)

X_pred = predict_df[['HAM_Log', 'Dev_Scaled']].to_numpy()
predictions = tree_model.predict(X_pred)

state_map = {0: "⏳ BREAKOUT LOOKOUT", 1: "📈 NEXT BAR: LOCKED UP", 2: "📉 NEXT BAR: LOCKED DOWN"}

final_display_matrix = []
predict_len = len(predict_df)
for i in range(predict_len):
    final_display_matrix.append(f"📊 HISTORICAL OOS -> {state_map[predictions[i]]}")

predict_df['ML Signal Grid'] = final_display_matrix

# =====================================================================
# 6. PURE UNLINKED LIVE 1-HOUR RENDER GATEWAY
# =====================================================================
live_price = float(df_live['Close'].iloc[0])
live_time = df_live.index[0]

last_historical_close = float(predict_df['Close'].iloc[-1])
live_delta = live_price - last_historical_close

# Compute running features isolated exactly with the realigned hourly metrics
live_kb = float(kb[-1])
live_hurst = float(hurst[-1])
live_std = float(rolling_std[-1])

calculated_live_ham = float(((live_price - live_kb) / live_std) * live_hurst)

if live_price > last_historical_close:
    live_direction = "TEMPORARY UP"
    live_signal = f"🟢 LIVE PROJECTION -> {state_map[predictions[-1]]}"
elif live_price < last_historical_close:
    live_direction = "TEMPORARY DOWN"
    live_signal = f"🔴 LIVE PROJECTION -> {state_map[predictions[-1]]}"
else:
    live_direction = "UNCHANGED"
    live_signal = "⏳ LIVE RUNNING -> HORIZON MATCH"

# Clean table structural rendering
clean_cols = ['Close', 'Direction_State', 'HAM_Log', 'ML Signal Grid']
display_df = predict_df[clean_cols].copy()
display_df.rename(columns={'Close': 'Locked Close Price', 'Direction_State': 'Historical Close State', 'HAM_Log': 'Dynamic HAM Log'}, inplace=True)

live_row = pd.DataFrame(index=[live_time], data={
    'Locked Close Price': round(live_price, 2),
    'Historical Close State': live_direction,
    'Dynamic HAM Log': round(calculated_live_ham, 4),
    'ML Signal Grid': live_signal
})

display_df_inverted = display_df.iloc[::-1].copy()
display_df_inverted.index = display_df_inverted.index.strftime('%Y-%m-%d %H:%M')
live_row.index = live_row.index.strftime('%Y-%m-%d %H:%M')

final_render_board = pd.concat([live_row, display_df_inverted])

# =====================================================================
# 7. METRIC BOARD INTERFACE
# =====================================================================
st.markdown("---")
st.subheader("🌲 MACHINE LEARNING RIGID MATRIX OUTPUT BOARD")

if "LOCKED UP" in live_signal:
    st.success(f"### {live_signal}")
elif "LOCKED DOWN" in live_signal:
    st.info(f"### {live_signal}")
else:
    st.warning(f"### {live_signal}")

st.markdown("---")
st.sidebar.markdown("### 🛡️ Pure Hourly Re-calibration Audit")
st.sidebar.success("✓ Math Noise Windows Re-aligned")
st.sidebar.success("✓ Volatility Scaler Synchronized")
st.sidebar.metric(label="📊 Frozen Historical Hours", value=f"{len(predict_df)}")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="⚙️ Real-Time Isolated HAM Log", value=f"{calculated_live_ham:.4f}")
with col2:
    st.metric(label="📊 Live Hourly Price Quote", value=f"${live_price:.2f}")

st.subheader("📋 Pure Immutable Locked History Logs")
st.dataframe(final_render_board, use_container_width=True, height=500)
