import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="BTC 1H Zero-Leak Radar", layout="wide")
st.title("⚡ BTC Standard 1-Hour Airtight Matrix")
st.write("🎯 **Pure 8-Step Verification:** 50:50 Data Split | 1H Rigid Candles | Zero Leak Dynamic Blocking.")

# Session Memory Block to store calculated and frozen historical states
if 'airtight_1h_diary' not in st.session_state:
    st.session_state['airtight_1h_diary'] = {}

# =====================================================================
# STEP 1 & 2: PURE MATHEMATICAL FEATURE ENGINES (ZERO LEAK FORWARD LOOK)
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
# STEP 3 & 4: 2-YEAR HISTORICAL 1-HOUR INGESTION WITH NO REPAINT LABELS
# =====================================================================
try:
    df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(subset=['Close'], inplace=True)
    
    # CRITICAL: STRICT FORWARD FILLING ONLY - NO BACKFILL ALLOWED TO PREVENT LEAK
    df = df.ffill()
except Exception as e:
    st.error(f"Ingestion System Malfunction: {e}")
    st.stop()

# Feature Calculations
close_arr = df['Close'].to_numpy(dtype=float)
df['Kalman'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
df['Hurst'] = calculate_rolling_hurst(close_arr, window=50)
df['HAM'] = (df['Close'] - df['Kalman']) * (df['Hurst'] * 2.0)

# STEP 5: PROPER 500-MULTIPLE STRIKE ENGINE
df['Near_ATM'] = np.round(df['Close'].to_numpy() / 500.0) * 500.0

# =====================================================================
# STEP 6: PURE 50:50 SEGREGATED PATTERN BOUNDARY
# =====================================================================
n_len = len(df)
split_idx = int(n_len * 0.50) # Absolute midpoint anchor

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

state_map = {0: "⚠️ RISK ZONE", 1: "👑 CE ZERO FIXED", 2: "👑 PE ZERO FIXED"}
final_outputs = ["⚠️ RISK ZONE"] * n_len

if len(train_df) > 50:
    prices_train = train_df['Close'].to_numpy()
    labels_train = np.zeros(len(train_df), dtype=int)
    
    # 24-step forward look ahead only inside historical boundaries for validation rules
    for idx in range(len(train_df) - 24):
        fut_settle = prices_train[idx + 24]
        current_atm = train_df['Near_ATM'].iloc[idx]
        if fut_settle < current_atm: labels_train[idx] = 1
        elif fut_settle > current_atm: labels_train[idx] = 2
        
    # Learning Matrix Mapping
    for idx in range(split_idx):
        final_outputs[idx] = state_map[labels_train[idx]]
        
    # Model Learning Phase (Only uses first 50% variables)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(train_df[['HAM', 'Near_ATM']].to_numpy(), labels_train)
    
    # Prediction Matrix Phase (Runs on last 50% out-of-sample data)
    test_preds = clf.predict(test_df[['HAM', 'Near_ATM']].to_numpy())
    for idx in range(len(test_df)):
        final_outputs[split_idx + idx] = state_map[test_preds[idx]]

# =====================================================================
# STEP 7: SESSION STORAGE OVERRIDE LAYER (NO BACK DATA RETRACTION)
# =====================================================================
final_fixed_matrix = []

for i in range(n_len):
    timestamp_key = df.index[i].strftime('%Y-%m-%d %H:%M')
    
    if i < split_idx:
        split_tag = "📚 LEARNING (FIRST 50%)"
    else:
        split_tag = "🔮 PREDICTION (LAST 50%)"
        
    if i == n_len - 1:
        split_tag = "🔄 LIVE TICKING ZONE"

    # Static snapshot validation check
    if i < n_len - 1:
        if timestamp_key in st.session_state['airtight_1h_diary']:
            saved_row = st.session_state['airtight_1h_diary'][timestamp_key]
            final_fixed_matrix.append(saved_row)
        else:
            atm_val = int(df['Near_ATM'].iloc[i])
            ham_val = float(df['HAM'].iloc[i])
            close_val = float(df['Close'].iloc[i])
            
            row_data = {
                'Time Axis': timestamp_key,
                'Split Phase': split_tag,
                'BTC Spot': close_val,
                'Dynamic HAM': ham_val,
                '🔒 Target Option Grid': f"{atm_val} | {final_outputs[i]} [🔒 RECORDED]"
            }
            st.session_state['airtight_1h_diary'][timestamp_key] = row_data
            final_fixed_matrix.append(row_data)
    else:
        # Live processing bar
        atm_val = int(df['Near_ATM'].iloc[i])
        ham_val = float(df['HAM'].iloc[i])
        close_val = float(df['Close'].iloc[i])
        
        live_row_data = {
            'Time Axis': timestamp_key,
            'Split Phase': split_tag,
            'BTC Spot': close_val,
            'Dynamic HAM': ham_val,
            '🔒 Target Option Grid': f"{atm_val} | {final_outputs[i]} [🔄 ACTIVE]"
        }
        final_fixed_matrix.append(live_row_data)

# =====================================================================
# STEP 8: INVERTED AUDIT LOG DISPLAY GRID
# =====================================================================
display_df = pd.DataFrame(final_fixed_matrix)
display_df.set_index('Time Axis', inplace=True)
display_df_inverted = display_df.iloc[::-1]

st.subheader("📋 Strict Mathematical Split & Execution Logs")
st.dataframe(display_df_inverted, use_container_width=True, height=600)
