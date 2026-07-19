import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="BTC 50:50 Visual Split", layout="wide")
st.title("⚡ BTC 200-Point Visible 50:50 Split Engine")
st.write("🎯 **Pure 8-Step Verification:** First 50% data is for LEARNING, Second 50% data is for PREDICTION.")

if 'rigid_split_diary' not in st.session_state:
    st.session_state['rigid_split_diary'] = {}

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
# 2. DATA INGESTION (2 YEARS - 1 HOUR)
# =====================================================================
try:
    df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(subset=['Close'], inplace=True)
    df = df.ffill()
except Exception as e:
    st.error(f"Ingestion Error: {e}")
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

# Feature Extraction
close_arr = df_range['Close'].to_numpy(dtype=float)
df_range['Kalman'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
df_range['Hurst'] = calculate_rolling_hurst(close_arr, window=50)
df_range['HAM'] = (df_range['Close'] - df_range['Kalman']) * (df_range['Hurst'] * 2.0)
df_range['Near_ATM'] = np.round(df_range['Close'].to_numpy() / 500.0) * 500.0

# =====================================================================
# 3. 🧠 VISIBLE 50:50 LEARNING VS PREDICTION SPLIT
# =====================================================================
n_len = len(df_range)
split_idx = int(n_len * 0.50) # Strict middle line

# Train Data (Pehle 50% bars jinse model seekhega)
train_df = df_range.iloc[:split_idx].copy()

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
    
    # Predict on entire dataset
    raw_preds = clf.predict(df_range[['HAM', 'Near_ATM']].to_numpy())
else:
    raw_preds = np.zeros(n_len, dtype=int)

state_map = {0: "⚠️ RISK ZONE", 1: "👑 CE ZERO FIXED", 2: "👑 PE ZERO FIXED"}

# =====================================================================
# 4. TRIPLE DEADBOLT SESSION DIARY WITH VISUAL LABELS
# =====================================================================
final_fixed_matrix = []

for i in range(n_len):
    timestamp_key = df_range.index[i].strftime('%Y-%m-%d %H:%M') + f"_idx_{i}"
    
    # Split Label Tagging
    if i < split_idx:
        split_tag = "📚 LEARNING ZONE (TRAIN)"
    else:
        split_tag = "🔮 OUT-OF-SAMPLE (TEST)"
        
    if i == n_len - 1:
        split_tag = "🔄 LIVE REAL-TIME"

    # Memory Check for historical rows
    if i < n_len - 1:
        if timestamp_key in st.session_state['rigid_split_diary']:
            saved_row = st.session_state['rigid_split_diary'][timestamp_key]
            final_fixed_matrix.append(saved_row)
        else:
            atm_val = int(df_range['Near_ATM'].iloc[i])
            ham_val = float(df_range['HAM'].iloc[i])
            close_val = float(df_range['Close'].iloc[i])
            status_text = state_map[raw_preds[i]]
            
            row_data = {
                'Time Stamp': timestamp_key.split('_idx_')[0],
                'Engine Split Type': split_tag,
                'BTC Price': close_val,
                'HAM': ham_val,
                '🔒 Locked Strike Reality': f"{atm_val} | {status_text} [🔒 FIXED]"
            }
            st.session_state['rigid_split_diary'][timestamp_key] = row_data
            final_fixed_matrix.append(row_data)
    else:
        # Live active row
        atm_val = int(df_range['Near_ATM'].iloc[i])
        ham_val = float(df_range['HAM'].iloc[i])
        close_val = float(df_range['Close'].iloc[i])
        status_text = state_map[raw_preds[i]]
        
        live_row_data = {
            'Time Stamp': timestamp_key.split('_idx_')[0],
            'Engine Split Type': split_tag,
            'BTC Price': close_val,
            'HAM': ham_val,
            '🔒 Locked Strike Reality': f"{atm_val} | {status_text} [🔄 LIVE]"
        }
        final_fixed_matrix.append(live_row_data)

# Grid Rendering
display_df = pd.DataFrame(final_fixed_matrix)
display_df.set_index('Time Stamp', inplace=True)
display_df_inverted = display_df.iloc[::-1]

st.subheader("📋 50:50 Segregated Audit Logs (Strict Memory Immutability)")
st.dataframe(display_df_inverted, use_container_width=True, height=600)
