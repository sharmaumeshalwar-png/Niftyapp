import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier

# Page Configuration
st.set_page_config(page_title="BTC ML Decision Tree Engine", layout="wide")
st.title("⚡ BTC 200-Point ML Pattern Learning Radar")
st.write("🎯 **Decision Tree Engine:** Pure 50:50 Learning vs Out-of-Sample Prediction Matrix Split.")

# =====================================================================
# 1. MATHEMATICAL ENGINES
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0: 
        return []
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
# 2. DATA INGESTION: 2 YEAR PERIOD @ 1 HOUR CANDLES
# =====================================================================
df = None
with st.spinner("Ingesting 2-Year Hourly Market History..."):
    try:
        df = yf.download(tickers="BTC-USD", period="2y", interval="1h", multi_level_index=False)
        if df is None or df.empty:
            st.error("🚨 Error: Data download failed from Yahoo Finance.")
            st.stop()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        if len(df) > 10: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close'], how='all', inplace=True)
            df = df.ffill() 
            df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_convert('Asia/Kolkata')
        else:
            st.error("🚨 Error: Insufficient data structure.")
            st.stop()
    except Exception as e:
        st.error(f"🚨 Ingestion Failure: {e}")
        st.stop()

# =====================================================================
# 3. 200-POINT RANGE CANDLE GENERATION
# =====================================================================
raw_closes = df['Close'].to_numpy(dtype=float)
raw_times = df.index
range_size = 200.0
range_closes, range_times = [raw_closes[0]], [raw_times[0]]
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

if raw_closes[-1] != range_closes[-1]:
    range_closes.append(raw_closes[-1])
    range_times.append(raw_times[-1])

df_range = pd.DataFrame(index=range_times, data={'Close': range_closes})
df_predict = df_range.copy()

# =====================================================================
# 4. FEATURE EXTRACTION
# =====================================================================
close_arr = df_predict['Close'].to_numpy(dtype=float)
df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
df_predict['Hurst'] = calculate_rolling_hurst(close_arr, window=50)

raw_weighted_momentum = df_predict['Close'] - df_predict['Kalman_Baseline']
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.to_numpy(), initial_p=0.50, q_val=0.001, r_val=0.1)
df_predict['Hurst_Amp_Momentum'] = df_predict['Weighted_Momentum'] * (df_predict['Hurst'] * 2.0)

df_predict['Normalized_Dev_Pct'] = (df_predict['Close'] - df_predict['Kalman_Baseline']) / df_predict['Kalman_Baseline']
df_predict.dropna(subset=['Hurst', 'Close', 'Normalized_Dev_Pct'], inplace=True)

strike_interval = 500.0
prices = df_predict['Close'].to_numpy()
n_len = len(df_predict)
near_atm_numeric = np.round(prices / strike_interval) * strike_interval
df_predict['Near_ATM'] = near_atm_numeric

# =====================================================================
# 5. ML TARGET LABEL GENERATION (EXPIRY RETROSPECT)
# =====================================================================
expiry_window = 750
target_labels = []

for idx in range(n_len):
    end_idx = min(idx + expiry_window, n_len - 1)
    future_max = np.max(prices[idx:end_idx + 1])
    future_min = np.min(prices[idx:end_idx + 1])
    future_settle = prices[end_idx]
    current_atm = near_atm_numeric[idx]
    
    if future_settle < current_atm and future_max < current_atm + 200:
        target_labels.append(1)  # CE Zero
    elif future_settle > current_atm and future_min > current_atm - 200:
        target_labels.append(2)  # PE Zero
    else:
        target_labels.append(0)  # Volatile

df_predict['Target_Class'] = target_labels

# =====================================================================
# 6. PURE 50:50 LEARN VS PREDICT ENGINE
# =====================================================================
# Pure mid-point matrix split calculated dynamically
mid_point = n_len // 2

# First 50%: Pure Training Set (with expiration window protection gap)
train_limit = max(10, mid_point - expiry_window)
train_df = df_predict.iloc[:train_limit].dropna()

# Last 50%: Pure Out-Of-Sample Prediction Show Matrix
predict_df = df_predict.iloc[mid_point:].copy()

if len(train_df) > 10 and len(predict_df) > 0:
    X_train = train_df[['Hurst_Amp_Momentum', 'Normalized_Dev_Pct']].to_numpy()
    y_train = train_df['Target_Class'].to_numpy()
    
    # Model configuration
    tree_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
    tree_model.fit(X_train, y_train)
    
    # Run prediction EXCLUSIVELY on the last 50% unseen data
    X_predict = predict_df[['Hurst_Amp_Momentum', 'Normalized_Dev_Pct']].to_numpy()
    predictions_out = tree_model.predict(X_predict)
else:
    predictions_out = np.zeros(len(predict_df))

state_map = {0: "⚠️ VOLATILE / BREACH EXPECTED", 1: "👑 CE ZERO FIXED", 2: "👑 PE ZERO FIXED"}

# Building display matrix strictly for the prediction half
final_display_matrix = []
predict_len = len(predict_df)
prices_predict = predict_df['Close'].to_numpy()
near_atm_predict = predict_df['Near_ATM'].to_numpy()

for i in range(predict_len):
    atm_str = str(int(near_atm_predict[i]))
    base_state = state_map[predictions_out[i]]
    
    # Dynamic live-edge flagging within the prediction sector
    if (mid_point + i) >= (n_len - expiry_window):
        final_display_matrix.append(f"Strike Ref: {atm_str} | 🟢 LIVE PROJECTION: {base_state.split(' ')[-1] if ' ' in base_state else base_state}")
    else:
        final_display_matrix.append(f"Strike Ref: {atm_str} | 📊 OUT-OF-SAMPLE TEST: {base_state}")

predict_df['🔒 Frozen Tree Decision'] = final_display_matrix

# =====================================================================
# 7. LIVE USER INTERFACE (SHOWING PREDICTION DATASET)
# =====================================================================
latest_row = predict_df.iloc[-1]

st.markdown("---")
st.subheader("🌲 MACHINE LEARNING DECISION TREE OUTPUT BOARD (PREDICTION HALF)")

if "CE ZERO" in latest_row['🔒 Frozen Tree Decision']:
    st.success(f"### {latest_row['🔒 Frozen Tree Decision']}")
elif "PE ZERO" in latest_row['🔒 Frozen Tree Decision']:
    st.info(f"### {latest_row['🔒 Frozen Tree Decision']}")
else:
    st.warning(f"### {latest_row['🔒 Frozen Tree Decision']}")

st.markdown("---")
st.sidebar.markdown(f"### 📊 Data Split Analytics")
st.sidebar.metric(label="📈 Total Generated Bars", value=f"{n_len}")
st.sidebar.metric(label="🧠 50% Learn Model Rows", value=f"{mid_point}")
st.sidebar.metric(label="👁️ 50% Show Predict Rows", value=f"{predict_len}")

r_col1, r_col2 = st.columns(2)
with r_col1:
    st.metric(label="⚙️ Real-Time Dynamic HAM", value=f"{latest_row['Hurst_Amp_Momentum']:.4f}")
with r_col2:
    st.metric(label="📊 BTC Spot Valuation", value=f"${latest_row['Close']:.2f}")

# Grid Matrix Visual for the Prediction Half
clean_cols = ['Close', 'Hurst_Amp_Momentum', '🔒 Frozen Tree Decision']
display_df = predict_df[clean_cols].copy()
display_df.rename(columns={
    'Close': 'BTC Price', 
    'Hurst_Amp_Momentum': '⚙️ Dynamic HAM Log',
    '🔒 Frozen Tree Decision': '🌲 ML Tree Learned Reality Grid'
}, inplace=True)

display_df_inverted = display_df.iloc[::-1].copy()
display_df_inverted.index = display_df_inverted.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Out-of-Sample Prediction Logs (Last 50% Matrix)")
st.dataframe(display_df_inverted, use_container_width=True, height=500)
