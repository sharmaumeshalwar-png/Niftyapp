import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Pure 50:50 Mathematical Split", layout="wide")
st.title("⚡ BTC Pure 50:50 Train-Predict Segregated Engine")
st.write("🎯 **Pure 8-Step Verification:** Mathematical boundary wall enabled. Zero interaction between train and test arrays.")

if 'absolute_split_diary' not in st.session_state:
    st.session_state['absolute_split_diary'] = {}

# =====================================================================
# 1. PURE MATHEMATICAL ENGINES (ZERO LEAK FORWARD LOOK)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0: return []
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
    if len(price_series) <= window: return hurst_values
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
# 2. DATA INGESTION & STRICT ZERO-BFILL POLICY
# =====================================================================
try:
    df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(subset=['Close'], inplace=True)
    df = df.ffill()  # Strictly forward fill, blocking bfill leak
except Exception as e:
    st.error(f"Data Fetch Error: {e}")
    st.stop()

# Array Conversion
close_arr = df['Close'].to_numpy(dtype=float)
time_arr = df.index

# Total Data Length & Strict Midpoint Split
n_total = len(close_arr)
midpoint = int(n_total * 0.50)

# =====================================================================
# 3. 📚 STRICT TRAINING ARRAY (FIRST 50% ONLY)
# =====================================================================
train_closes = close_arr[:midpoint]
train_kalman = apply_kalman_filter_custom(train_closes, initial_p=50.0, q_val=0.0005, r_val=0.2)
train_hurst = calculate_rolling_hurst(train_closes, window=50)
train_ham = (train_closes - train_kalman) * (train_hurst * 2.0)
train_atm = np.round(train_closes / 500.0) * 500.0

# Training Labels (Target Scan restricted only inside train array boundaries)
labels_train = np.zeros(len(train_closes), dtype=int)
for idx in range(len(train_closes) - 24):
    fut_settle = train_closes[idx + 24]
    if fut_settle < train_atm[idx]: labels_train[idx] = 1
    elif fut_settle > train_atm[idx]: labels_train[idx] = 2

# Fit Classifier ONLY on Train Features
X_train = np.column_stack((train_ham, train_atm))
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, labels_train)

# =====================================================================
# 4. 🔮 PURE OUT-OF-SAMPLE TEST ARRAY (LAST 50% ONLY)
# =====================================================================
test_closes = close_arr[midpoint:]
test_kalman = apply_kalman_filter_custom(test_closes, initial_p=50.0, q_val=0.0005, r_val=0.2)
test_hurst = calculate_rolling_hurst(test_closes, window=50)
test_ham = (test_closes - test_kalman) * (test_hurst * 2.0)
test_atm = np.round(test_closes / 500.0) * 500.0

X_test = np.column_stack((test_ham, test_atm))
test_preds = clf.predict(X_test)

# =====================================================================
# 5. MATRIX COMPOSITION & DIARY LOCK
# =====================================================================
state_map = {0: "⚠️ RISK ZONE", 1: "👑 CE ZERO FIXED", 2: "👑 PE ZERO FIXED"}
final_matrix_list = []

for i in range(n_total):
    timestamp_key = time_arr[i].strftime('%Y-%m-%d %H:%M')
    
    # Check if index belongs to Train or Test section
    if i < midpoint:
        split_phase = "📚 LEARNING (FIRST 50%)"
        price_val = train_closes[i]
        ham_val = train_ham[i]
        atm_val = int(train_atm[i])
        status_text = state_map[labels_train[i]]
    else:
        split_phase = "🔮 PREDICTION (LAST 50%)"
        test_idx = i - midpoint
        price_val = test_closes[test_idx]
        ham_val = test_ham[test_idx]
        atm_val = int(test_atm[test_idx])
        status_text = state_map[test_preds[test_idx]]
        
    if i == n_total - 1:
        split_phase = "🔄 LIVE REAL-TIME LAYER"

    # Lock into memory structure to prevent repainting
    if i < n_total - 1:
        if timestamp_key in st.session_state['absolute_split_diary']:
            saved_row = st.session_state['absolute_split_diary'][timestamp_key]
            final_fixed_matrix_row = saved_row
        else:
            row_data = {
                'Time Axis': timestamp_key,
                'Split Phase': split_phase,
                'BTC Spot': price_val,
                'Dynamic HAM': ham_val,
                '🔒 Locked Core Reality': f"{atm_val} | {status_text} [🔒 RECORDED]"
            }
            st.session_state['absolute_split_diary'][timestamp_key] = row_data
            final_fixed_matrix_row = row_data
    else:
        # Keep the final row dynamically tracking
        final_fixed_matrix_row = {
            'Time Axis': timestamp_key,
            'Split Phase': split_phase,
            'BTC Spot': price_val,
            'Dynamic HAM': ham_val,
            '🔒 Locked Core Reality': f"{atm_val} | {status_text} [🔄 ACTIVE]"
        }
        
    final_matrix_list.append(final_fixed_matrix_row)

# Inverted Layout Grid Display
display_df = pd.DataFrame(final_matrix_list)
display_df.set_index('Time Axis', inplace=True)
display_df_inverted = display_df.iloc[::-1]

st.subheader("📋 Verification Matrix (Strict Independent Arrays Checked)")
st.dataframe(display_df_inverted, use_container_width=True, height=600)
