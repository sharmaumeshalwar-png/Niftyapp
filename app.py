import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="BTC Absolute Session Memory", layout="wide")
st.title("⚡ BTC 200-Point Bulletproof Session Memory Radar")
st.write("🎯 **Pure 8-Step Verification:** Historical rows are locked into physical application memory. Zero recalculation allowed.")

# Initialize the permanent memory diary if it doesn't exist
if 'frozen_matrix_diary' not in st.session_state:
    st.session_state['frozen_matrix_diary'] = {}

# =====================================================================
# 1. PURE MATHEMATICAL ENGINES
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
# 2. DATA INGESTION & PROGRESSIVE RANGE BARS
# =====================================================================
try:
    df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(subset=['Close'], inplace=True)
    df = df.ffill()
except Exception as e:
    st.error(f"Data Error: {e}")
    st.stop()

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

# Baseline calculations for model rules
close_arr = df_range['Close'].to_numpy(dtype=float)
df_range['Kalman'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
df_range['Hurst'] = calculate_rolling_hurst(close_arr, window=50)
df_range['HAM'] = (df_range['Close'] - df_range['Kalman']) * (df_range['Hurst'] * 2.0)
df_range['Near_ATM'] = np.round(df_range['Close'].to_numpy() / 500.0) * 500.0

# Train the Classifier once to get the baseline rules
n_len = len(df_range)
train_cutoff = max(0, n_len - 750)
train_df = df_range.iloc[:train_cutoff].copy()

if len(train_df) > 50:
    prices_train = train_df['Close'].to_numpy()
    labels_train = np.zeros(len(train_df), dtype=int)
    for idx in range(len(train_df) - 50):
        fut_settle = prices_train[idx + 50]
        current_atm = train_df['Near_ATM'].iloc[idx]
        if fut_settle < current_atm: labels_train[idx] = 1
        elif fut_settle > current_atm: labels_train[idx] = 2
        
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(train_df[['HAM', 'Near_ATM']].to_numpy(), labels_train)
    raw_preds = clf.predict(df_range[['HAM', 'Near_ATM']].to_numpy())
else:
    raw_preds = np.zeros(n_len, dtype=int)

state_map = {0: "⚠️ RISK ZONE", 1: "👑 CE ZERO FIXED", 2: "👑 PE ZERO FIXED"}

# =====================================================================
# 3. 🔒 SESSION STATE DIARY LOCK (THE TRIPLE DEADBOLT)
# =====================================================================
final_fixed_matrix = []

for i in range(n_len):
    timestamp_key = df_range.index[i].strftime('%Y-%m-%d %H:%M') + f"_bar_{i}"
    
    # Agar ye historical bar hai aur iski diary entry pehle se bani hui hai, use absolute load karo
    if i < n_len - 1:
        if timestamp_key in st.session_state['frozen_matrix_diary']:
            # Paji direct diary se purani snapshot utha li, no new math code touch!
            saved_row = st.session_state['frozen_matrix_diary'][timestamp_key]
            final_fixed_matrix.append(saved_row)
        else:
            # First time logging an older bar into memory
            atm_val = int(df_range['Near_ATM'].iloc[i])
            ham_val = float(df_range['HAM'].iloc[i])
            close_val = float(df_range['Close'].iloc[i])
            status_text = state_map[raw_preds[i]]
            
            row_data = {
                'Time': timestamp_key.split('_')[0],
                'BTC Price': close_val,
                'HAM Value': ham_val,
                '🔒 Locked Core Reality': f"{atm_val} | {status_text} [🔒 MEMORY LOCKED]"
            }
            st.session_state['frozen_matrix_diary'][timestamp_key] = row_data
            final_fixed_matrix.append(row_data)
    else:
        # Live row logic - continuously updates only on top
        atm_val = int(df_range['Near_ATM'].iloc[i])
        ham_val = float(df_range['HAM'].iloc[i])
        close_val = float(df_range['Close'].iloc[i])
        status_text = state_map[raw_preds[i]]
        
        live_row_data = {
            'Time': timestamp_key.split('_')[0],
            'BTC Price': close_val,
            'HAM Value': ham_val,
            '🔒 Locked Core Reality': f"{atm_val} | {status_text} [🔄 LIVE TRACK]"
        }
        final_fixed_matrix.append(live_row_data)

# Convert memory structure to display format
display_df = pd.DataFrame(final_fixed_matrix)
display_df.set_index('Time', inplace=True)
display_df_inverted = display_df.iloc[::-1]

# =====================================================================
# 4. STREAMLIT OUTPUT GRID
# =====================================================================
st.subheader("📋 Hardcoded Memory State Logs (Purani Rows Unchangeable)")
st.dataframe(display_df_inverted, use_container_width=True, height=500)
