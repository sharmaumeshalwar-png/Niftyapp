import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# STEP 1: INITIALIZE AIRTIGHT INTERFACE WAREHOUSE
st.set_page_config(page_title="Rigid 2024-2026 Date Engine", layout="wide")
st.title("⚡ BTC 2-Year Matrix: Explicit Date Alignment Timeline")
st.write("🎯 **Pure 8-Step Verification:** Live July 2026 Horizon | 50:50 Clean Split Verification.")

# STEP 2: MATHEMATICAL MATH ENGINE DEFINITIONS
def apply_kalman_filter_custom(data_array):
    if len(data_array) == 0: return np.array([])
    x, p = data_array[0], 50.0  
    q_val, r_val = 0.0005, 0.2
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
    if len(price_series) <= window: return list(hurst_values)
    log_returns = np.log(price_series / np.roll(price_series, 1))
    log_returns[0] = 0
    for i in range(window, len(price_series)):
        window_data = log_returns[i-window+1:i+1]
        cum_dev = np.cumsum(window_data - np.mean(window_data))
        r_val = np.max(cum_dev) - np.min(cum_dev)
        s_val = np.std(window_data) + 1e-10
        hurst_values[i] = np.clip(r_val / s_val, 0.0, 1.0)
    return list(hurst_values)

# STEP 3: DATA INGESTION (FETCHES EXACTLY UP TO 2026 LIVE HORIZON)
try:
    raw_df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = [col[0] for col in raw_df.columns]
        
    df = raw_df[['Close']].copy()
    df.dropna(subset=['Close'], inplace=True)
    df = df.ffill() 
except Exception as e:
    st.error(f"Ingestion Fault: {e}")
    st.stop()

# STEP 4: PHYSICAL NUMPY ARRAY EXTRACTION & MIDPOINT ANCHOR
raw_closes = df['Close'].to_numpy(dtype=float)
timestamps = df.index
n_total = len(raw_closes)

midpoint_idx = int(n_total * 0.50)

# STEP 5: 📚 NUMPY TRAINING BOUNDARY (FIRST 50% - PAST TIMELINE)
train_closes = raw_closes[:midpoint_idx]
train_kalman = apply_kalman_filter_custom(train_closes)
train_hurst = np.array(calculate_rolling_hurst(train_closes))
train_ham = (train_closes - train_kalman) * (train_hurst * 2.0)
train_atm = np.round(train_closes / 500.0) * 500.0
train_velocity = np.zeros(len(train_closes))
train_velocity[1:] = np.diff(train_closes) 

labels_train = np.zeros(len(train_closes), dtype=int)
for idx in range(len(train_closes) - 24):
    if train_closes[idx + 24] < train_atm[idx]: 
        labels_train[idx] = 1
    elif train_closes[idx + 24] > train_atm[idx]: 
        labels_train[idx] = 2

X_train = np.column_stack((train_ham, train_atm, train_hurst, train_velocity))
clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
clf.fit(X_train, labels_train)

# STEP 6: 🔮 NUMPY PREDICTION BOUNDARY (LAST 50% - FORWARD UP TO CURRENT 2026)
test_closes = raw_closes[midpoint_idx:]
test_kalman = apply_kalman_filter_custom(test_closes)
test_hurst = np.array(calculate_rolling_hurst(test_closes))
test_ham = (test_closes - test_kalman) * (test_hurst * 2.0)
test_atm = np.round(test_closes / 500.0) * 500.0
test_velocity = np.zeros(len(test_closes))
test_velocity[1:] = np.diff(test_closes)

X_test = np.column_stack((test_ham, test_atm, test_hurst, test_velocity))
test_predictions = clf.predict(X_test)

# STEP 7: MEMORY SNAPSHOT SEPARATION STRATEGY
state_map = {0: "⚠️ RISK ZONE", 1: "👑 CE ZERO FIXED", 2: "👑 PE ZERO FIXED"}
train_matrix = []
test_matrix = []

for i in range(n_total):
    ts_key = timestamps[i].strftime('%Y-%m-%d %H:%M')
    
    if i < midpoint_idx:
        row_structure = {
            'Time Axis': ts_key,
            'Split Phase': "📚 LEARNING (PAST)",
            'BTC Spot': round(float(train_closes[i]), 2),
            'Dynamic HAM': round(float(train_ham[i]), 4),
            '🔒 Core Grid Result': f"{int(train_atm[i])} | {state_map[labels_train[i]]}"
        }
        train_matrix.append(row_structure)
    else:
        t_idx = i - midpoint_idx
        live_tag = "[🔄 LIVE HORIZON]" if i == n_total - 1 else "[🔒 FORWARD PREDICTED]"
        row_structure = {
            'Time Axis': ts_key,
            'Split Phase': "🔮 PREDICTION (FORWARD)",
            'BTC Spot': round(float(test_closes[t_idx]), 2),
            'Dynamic HAM': round(float(test_ham[t_idx]), 4),
            '🔒 Core Grid Result': f"{int(test_atm[t_idx])} | {state_map[test_predictions[t_idx]]} {live_tag}"
        }
        test_matrix.append(row_structure)

train_df = pd.DataFrame(train_matrix).set_index('Time Axis')
test_df = pd.DataFrame(test_matrix).set_index('Time Axis')

# STEP 8: EXPLICIT TIMELINE VISUALIZATION WITH STREAMLIT METRICS
start_date_overall = timestamps[0].strftime('%B %Y')
mid_date_overall = timestamps[midpoint_idx].strftime('%B %Y')
end_date_overall = timestamps[-1].strftime('%B %Y')

col1, col2, col3 = st.columns(3)
col1.metric("📅 Timeline Start", start_date_overall, "Data Ingestion Point")
col2.metric("✂️ Physical 50:50 Cutoff", mid_date_overall, "Model Shift Point")
col3.metric("🚀 Current Live State", end_date_overall, "July 2026 Engine Edge")

tab1, tab2 = st.tabs(["📚 Historical Learning Domain", "🔮 Live Evaluation Domain"])

with tab1:
    st.subheader(f"📋 Training Grid Target ({start_date_overall} ➡️ {mid_date_overall})")
    st.caption("Yeh model ka historical database hai jisse rule patterns sikhe gaye hain.")
    st.dataframe(train_df.iloc[::-1], use_container_width=True, height=450)

with tab2:
    st.subheader(f"📋 Prediction Grid Target ({mid_date_overall} ➡️ {end_date_overall})")
    st.caption("Yeh un-seen forward timeline hai jisme continuous real-time execution chal raha hai.")
    st.dataframe(test_df.iloc[::-1], use_container_width=True, height=450)
