import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Conditional Freeze Engine", layout="wide")
st.title("⚡ BTC 200-Point Range Bar Conditionally Frozen Radar")
st.write("🎯 **8-Step Dynamic HAM Freeze Engine:** Strike price stays dynamic until HAM hits the stabilization zone, then locks solid down to the second last row.")

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
# 2. SYSTEM DATA INGESTION
# =====================================================================
df = None
with st.spinner("Fetching Live Bitcoin Data..."):
    try:
        df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
        if df is None or df.empty:
            df = yf.download(tickers="BTC-USD", period="max", interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if len(df) > 10: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            df = df.ffill().bfill()
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_convert('Asia/Kolkata')
        else:
            st.error("🚨 Error: Insufficient data from API.")
            st.stop()
    except Exception as e:
        st.error(f"🚨 API Failure: {e}")
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
df_predict = df_range.iloc[int(len(df_range) * 0.50):].copy() if len(df_range) > 100 else df_range.copy()

# =====================================================================
# 4. LIVE DYNAMIC HAM VALUES CALCULATION
# =====================================================================
close_arr = df_predict['Close'].to_numpy(dtype=float)
df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr)
df_predict['Hurst'] = calculate_rolling_hurst(close_arr)
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom((df_predict['Close'] - df_predict['Kalman_Baseline']).to_numpy())

# HAM stays completely dynamic here paji!
df_predict['Hurst_Amp_Momentum'] = df_predict['Weighted_Momentum'] * (df_predict['Hurst'] * 2.0)
df_predict.dropna(subset=['Hurst', 'Close'], inplace=True)

strike_interval = 500.0
prices = df_predict['Close'].to_numpy()
n_len = len(df_predict)
near_atm_numeric = np.round(prices / strike_interval) * strike_interval

# =====================================================================
# 5. 🔒 CONDITION-BASED STRIKE FREEZING RADAR
# =====================================================================
frozen_strikes = []
last_valid_frozen_strike = "⚠️ SCANNING MOMENTUM"
is_frozen = False

for i in range(n_len):
    ham_value = df_predict['Hurst_Amp_Momentum'].iloc[i]
    current_atm_strike = str(int(near_atm_numeric[i]))
    
    # CONDITION: Agar HAM threshold break karke fix ho raha hai (>0.15 ya <-0.15) 
    # Aur system pehle se freeze nahi hai, toh strike lock kar do!
    if abs(ham_value) >= 0.15 and not is_frozen:
        if ham_value > 0.15:
            last_valid_frozen_strike = f"❄️ FROZEN ZERO PE: {current_atm_strike} PE"
        else:
            last_valid_frozen_strike = f"❄️ FROZEN ZERO CE: {current_atm_strike} CE"
        is_frozen = True
    
    # Agar HAM wapas normal cooling zone me aata hai, toh freeze release krdo taaki dynamic ho jaye
    elif abs(ham_value) < 0.05 and is_frozen:
        is_frozen = False
        
    # Agar freeze state active hai, toh purani locked value hi print hogi cell me (No repainting)
    if is_frozen:
        frozen_strikes.append(last_valid_frozen_strike)
    else:
        # Jab tak dynamic hai, tab tak live update show karega
        frozen_strikes.append(f"🔄 DYNAMIC ATM: {current_atm_strike}")

df_predict['Conditional_Strike_State'] = frozen_strikes

# =====================================================================
# 6. LIVE VISUAL RADAR
# =====================================================================
latest_row = df_predict.iloc[-1]

st.markdown("---")
st.subheader("🎯 PAJI LIVE RADAR: DYNAMIC HAM + CONDITIONALLY FROZEN STRIKE")

# Main Strike status container box
if "❄️" in latest_row['Conditional_Strike_State']:
    st.success(f"### {latest_row['Conditional_Strike_State']}")
    st.info("💡 **Status:** HAM value stabilize ho chuki hai! Strike price ko successfully freeze kar diya gaya hai.")
else:
    st.warning(f"### {latest_row['Conditional_Strike_State']}")
    st.info("💡 **Status:** HAM value abhi fluctuation range me hai, isliye Strike dynamic update ho rahi hai.")

st.markdown("---")

h_col1, h_col2 = st.columns(2)
with h_col1:
    st.metric(label="⚙️ Real-Time Dynamic HAM Value", value=f"{latest_row['Hurst_Amp_Momentum']:.4f}")
with h_col2:
    st.metric(label="📊 Live BTC Spot Value", value=f"${latest_row['Close']:.2f}")

# =====================================================================
# 7. GRID MATRIX FOR CONDITIONAL ANALYSIS
# =====================================================================
clean_cols = ['Close', 'Hurst_Amp_Momentum', 'Conditional_Strike_State']
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={
    'Close': 'BTC Price', 
    'Hurst_Amp_Momentum': '⚙️ Dynamic HAM Log',
    'Conditional_Strike_State': '🔒 Strike Price Matrix State (Second Last Layer Ready)'
}, inplace=True)

display_df_inverted = display_df.iloc[::-1].copy()
display_df_inverted.index = display_df_inverted.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Core Engine Log Sheet")
st.dataframe(display_df_inverted, use_container_width=True, height=500)
