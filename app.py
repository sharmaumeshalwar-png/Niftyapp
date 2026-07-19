import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier

# Page Configuration
st.set_page_config(page_title="BTC ML Decision Tree Engine", layout="wide")
st.title("⚡ BTC 200-Point ML Pattern Learning Radar")
st.write("🎯 **Decision Tree Engine:** Learning from historical HAM patterns to predict verified month-end zero strikes without look-ahead leaks.")

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
# 2. FIXED LEAK-PROOF DATA INGESTION
# =====================================================================
df = None
with st.spinner("Ingesting & Cleaning Market History..."):
    try:
        df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
        if df is None or df.empty:
            df = yf.download(tickers="BTC-USD", period="max", interval="1d")
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
# 4. FEATURE EXTRACTION (HAM & NEAR ATM)
# =====================================================================
close_arr = df_predict['Close'].to_numpy(dtype=float)
df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
df_predict['Hurst'] = calculate_rolling_hurst(close_arr, window=50)

raw_weighted_momentum = df_predict['Close'] - df_predict['Kalman_Baseline']
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.to_numpy(), initial_p=0.50, q_val=0.001, r_val=0.1)
df_predict['Hurst_Amp_Momentum'] = df_predict['Weighted_Momentum'] * (df_predict['Hurst'] * 2.0)
df_predict.dropna(subset=['Hurst', 'Close'], inplace=True)

strike_interval = 500.0
prices = df_predict['Close'].to_numpy()
n_len = len(df_predict)
near_atm_numeric = np.round(prices / strike_interval) * strike_interval
df_predict['Near_ATM'] = near_atm_numeric

# =====================================================================
# 5. ML TARGET LABEL GENERATION (PAST MONTH-END EXPIRY REALITIES)
# =====================================================================
expiry_window = 750
target_labels = []

for idx in range(n_len):
    end_idx = min(idx + expiry_window, n_len - 1)
    future_max = np.max(prices[idx:end_idx + 1])
    future_min = np.min(prices[idx:end_idx + 1])
    future_settle = prices[end_idx]
    current_atm = near_atm_numeric[idx]
    
    # Label mapping based on actual outcomes
    if future_settle < current_atm and future_max < current_atm + 200:
        target_labels.append(1)  # Verified CE Zero Zone
    elif future_settle > current_atm and future_min > current_atm - 200:
        target_labels.append(2)  # Verified PE Zero Zone
    else:
        target_labels.append(0)  # Volatile/Breached Zone

df_predict['Target_Class'] = target_labels

# =====================================================================
# 6. TRAINING THE ML DECISION TREE (LEARNING FROM PATTERNS)
# =====================================================================
# Train model using only historical bars (excluding the very end to prevent bleed)
train_df = df_predict.iloc[:-expiry_window].dropna() if n_len > expiry_window else df_predict.dropna()

if len(train_df) > 10:
    X_train = train_df[['Hurst_Amp_Momentum', 'Near_ATM']].to_numpy()
    y_train = train_df['Target_Class'].to_numpy()
    
    # Strict Decision Tree Classifier
    tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_model.fit(X_train, y_train)
    
    # Predict for the entire matrix based on pure historical tree mapping
    X_all = df_predict[['Hurst_Amp_Momentum', 'Near_ATM']].to_numpy()
    predictions = tree_model.predict(X_all)
else:
    predictions = np.zeros(n_len)

# Map predictions back to descriptive string representations
state_map = {0: "⚠️ VOLATILE / BREACH EXPECTED", 1: "👑 PAST PATTERN: CE ZERO FIXED", 2: "👑 PAST PATTERN: PE ZERO FIXED"}
df_predict['Tree_Learned_State'] = [state_map[p] for p in predictions]

# =====================================================================
# 7. 🔒 FREEZE MATRIX (SECOND LAST ROW IMPLEMENTATION)
# =====================================================================
final_display_matrix = []
for i in range(n_len):
    atm_str = str(int(near_atm_numeric[i]))
    pattern_status = df_predict['Tree_Learned_State'].iloc[i]
    final_display_matrix.append(f"Strike Ref: {atm_str} | {pattern_status}")

df_predict['🔒 Frozen Tree Decision'] = final_display_matrix

# =====================================================================
# 8. LIVE USER INTERFACE
# =====================================================================
latest_row = df_predict.iloc[-1]

st.markdown("---")
st.subheader("🌲 MACHINE LEARNING DECISION TREE OUTPUT BOARD")

if "CE ZERO" in latest_row['🔒 Frozen Tree Decision']:
    st.success(f"### {latest_row['🔒 Frozen Tree Decision']}")
elif "PE ZERO" in latest_row['🔒 Frozen Tree Decision']:
    st.info(f"### {latest_row['🔒 Frozen Tree Decision']}")
else:
    st.warning(f"### {latest_row['🔒 Frozen Tree Decision']}")

st.markdown("---")
r_col1, r_col2 = st.columns(2)
with r_col1:
    st.metric(label="⚙️ Real-Time Dynamic HAM", value=f"{latest_row['Hurst_Amp_Momentum']:.4f}")
with r_col2:
    st.metric(label="📊 BTC Spot Valuation", value=f"${latest_row['Close']:.2f}")

# Grid Matrix Visual
clean_cols = ['Close', 'Hurst_Amp_Momentum', '🔒 Frozen Tree Decision']
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={
    'Close': 'BTC Price', 
    'Hurst_Amp_Momentum': '⚙️ Dynamic HAM Log',
    '🔒 Frozen Tree Decision': '🌲 ML Tree Learned Reality Grid'
}, inplace=True)

display_df_inverted = display_df.iloc[::-1].copy()
display_df_inverted.index = display_df_inverted.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Tree Node History Logs (Second Last Layer Frozen)")
st.dataframe(display_df_inverted, use_container_width=True, height=500)
