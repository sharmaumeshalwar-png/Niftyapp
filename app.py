import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier

# STEP 1: INITIALIZE HARD WAREHOUSE ARCHITECTURE
st.set_page_config(page_title="Rigid 50:50 Verification Engine", layout="wide")
st.title("⚡ BTC Strict 50:50 Structural Split Matrix")
st.write("🎯 **Pure 8-Step Verification:** 1H Fixed Indexing | Zero-Leak Array Boundaries | Isolated Prediction Grid.")

if 'absolute_frozen_diary' not in st.session_state:
    st.session_state['absolute_frozen_diary'] = {}

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
    if len(price_series) <= window: return hurst_values
    log_returns = np.log(price_series / np.roll(price_series, 1))
    log_returns[0] = 0
    for i in range(window, len(price_series)):
        window_data = log_returns[i-window+1:i+1]
        cum_dev = np.cumsum(window_data - np.mean(window_data))
        r_val = np.max(cum_dev) - np.min(cum_dev)
        s_val = np.std(window_data) + 1e-10
        hurst_values[i] = np.clip(r_val / s_val, 0.0, 1.0)
    return hurst_values

# STEP 3: DATA INGESTION (2-YEAR FIXED 1-HOUR TIMESTAMPS)
try:
    df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(subset=['Close'], inplace=True)
    df = df.ffill()  # Pure forward fill to block bfill leakage
except Exception as e:
    st.error(f"Ingestion Fault: {e}")
    st.stop()

# STEP 4: STRICT ARRAY BOUNDARY EXTRACTION
raw_closes = df['Close'].to_numpy(dtype=float)
timestamps = df.index
n_total = len(raw_closes)

# Absolute Midpoint Physical Split (Force exact 50% split count)
midpoint_idx = int(n_total * 0.50)

# STEP 5:📚 ISOLATED TRAINING DOMAIN (FIRST 50% ARRAY ONLY)
train_closes = raw_closes[:midpoint_idx]
train_kalman = apply_kalman_filter_custom(train_closes)
train_hurst = calculate_rolling_hurst(train_closes)
train_ham = (train_closes - train_kalman) * (train_hurst * 2.0)
train_atm = np.round(train_closes / 500.0) * 500.0

# Train Target Label Generation
labels_train = np.zeros(len(train_closes), dtype=int)
for idx in range(len(train_closes) - 24):
    if train_closes[idx + 24] < train_atm[idx]: 
        labels_train[idx] = 1
    elif train_closes[idx + 24] > train_atm[idx]: 
        labels_train[idx] = 2

# Fit Classifier ONLY on First 50% Features
X_train = np.column_stack((train_ham, train_atm))
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, labels_train)

# STEP 6: 🔮 ISOLATED PREDICTION DOMAIN (LAST 50% ARRAY ONLY)
test_closes = raw_closes[midpoint_idx:]
test_kalman = apply_kalman_filter_custom(test_closes)
test_hurst = calculate_rolling_hurst(test_closes)
test_ham = (test_closes - test_kalman) * (test_hurst * 2.0)
test_atm = np.round(test_closes / 500.0) * 500.0

X_test = np.column_stack((test_ham, test_atm))
test_predictions = clf.predict(X_test)

# STEP 7: COMPOSITE CONSOL DATA LOGGING (ZERO REPAINT HARDWARE)
state_map = {0: "⚠️ RISK ZONE", 1: "👑 CE ZERO FIXED", 2: "👑 PE ZERO FIXED"}
final_matrix = []

for i in range(n_total):
    ts_key = timestamps[i].strftime('%Y-%m-%d %H:%M')
    
    # Check physical index placement to allocate exact split tags
    if i < midpoint_idx:
        split_phase = "📚 LEARNING (FIRST 50%)"
        spot_price = train_closes[i]
        ham_metric = train_ham[i]
        atm_strike = int(train_atm[i])
        engine_verdict = state_map[labels_train[i]]
    else:
        split_phase = "🔮 PREDICTION (LAST 50%)"
        t_idx = i - midpoint_idx
        spot_price = test_closes[t_idx]
        ham_metric = test_ham[t_idx]
        atm_strike = int(test_atm[t_idx])
        engine_verdict = state_map[test_predictions[t_idx]]
        
    if i == n_total - 1:
        split_phase = "🔄 LIVE TICKING LAYER"

    # Memory state storage injection
    if i < n_total - 1:
        if ts_key in st.session_state['absolute_frozen_diary']:
            final_matrix.append(st.session_state['absolute_frozen_diary'][ts_key])
        else:
            row_structure = {
                'Time Axis': ts_key,
                'Split Phase': split_phase,
                'BTC Spot': spot_price,
                'Dynamic HAM': ham_metric,
                '🔒 Target Option Grid': f"{atm_strike} | {engine_verdict} [🔒 FIXED]"
            }
            st.session_state['absolute_frozen_diary'][ts_key] = row_structure
            final_matrix.append(row_structure)
    else:
        # Dynamic live candle row allocation
        final_matrix.append({
            'Time Axis': ts_key,
            'Split Phase': split_phase,
            'BTC Spot': spot_price,
            'Dynamic HAM': ham_metric,
            '🔒 Target Option Grid': f"{atm_strike} | {engine_verdict} [🔄 LIVE]"
        })

# STEP 8: INVERTED FINAL GRID RENDER
display_dataframe = pd.DataFrame(final_matrix)
display_dataframe.set_index('Time Axis', inplace=True)
st.subheader("📋 Final Strict Validated Verification Grid")
st.dataframe(display_dataframe.iloc[::-1], use_container_width=True, height=600)
