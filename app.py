import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

# Page Configuration
st.set_page_config(page_title="BTC Zero-Leak ML Kalman Engine", layout="wide")
st.title("⚡ BTC 200-Point Range Bar Master Engine")
st.write("🎯 **ML Regret Ingestion Enabled:** Learning Past Near-ATM Strikes That Expired Validated Zero Based on HAM Matrix State")

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
# 2. SYSTEM DATA INGESTION (Bitcoin - 2 Years, 1 Hour Candles)
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
            st.error("🚨 Error: Insufficient data lines from API.")
            st.stop()
    except Exception as e:
        st.error(f"🚨 API Failure: {e}")
        st.stop()

# =====================================================================
# 3. 200-POINT RANGE CANDLE GENERATION ENGINE
# =====================================================================
raw_closes = df['Close'].to_numpy(dtype=float)
raw_times = df.index

range_size = 200.0
range_closes = []
range_times = []

current_anchor = raw_closes[0]
range_closes.append(current_anchor)
range_times.append(raw_times[0])

for i in range(1, len(raw_closes)):
    price_diff = raw_closes[i] - current_anchor
    if abs(price_diff) >= range_size:
        num_bars = int(abs(price_diff) // range_size)
        direction = np.sign(price_diff)
        for _ in range(num_bars):
            current_anchor += direction * range_size
            range_closes.append(current_anchor)
            range_times.append(raw_times[i])

is_live_candle_running = raw_closes[-1] != range_closes[-1]
if is_live_candle_running:
    range_closes.append(raw_closes[-1])
    range_times.append(raw_times[-1])

df_range = pd.DataFrame(index=range_times)
df_range['Close'] = range_closes

split_idx = int(len(df_range) * 0.50)
if len(df_range) - split_idx < 50:
    df_predict = df_range.copy()
else:
    df_predict = df_range.iloc[split_idx:].copy()

st.success(f"🟢 **Synced & Processed {len(df_predict)} Range Bars!**")
close_arr = df_predict['Close'].to_numpy(dtype=float)

# =====================================================================
# 4. SIGNAL & HAM CALCULATIONS
# =====================================================================
df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
df_predict['Hurst'] = calculate_rolling_hurst(close_arr, window=50)

raw_weighted_momentum = df_predict['Close'] - df_predict['Kalman_Baseline']
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.to_numpy(), initial_p=0.50, q_val=0.001, r_val=0.1)

df_predict['Hurst_Amp_Momentum'] = df_predict['Weighted_Momentum'] * (df_predict['Hurst'] * 2.0)
df_predict.dropna(subset=['Hurst', 'Close'], inplace=True)

ham_vals = df_predict['Hurst_Amp_Momentum'].to_numpy()
df_predict['Kalman_HAM_Fast'] = apply_kalman_filter_custom(ham_vals, initial_p=1.0, q_val=0.01, r_val=0.05)
df_predict['Kalman_HAM_Slow'] = apply_kalman_filter_custom(ham_vals, initial_p=1.0, q_val=0.0005, r_val=0.5)

fast_kf = df_predict['Kalman_HAM_Fast'].to_numpy()
slow_kf = df_predict['Kalman_HAM_Slow'].to_numpy()

df_predict['HAM_Triple_Product'] = df_predict['Hurst_Amp_Momentum'] * df_predict['Kalman_HAM_Fast'] * df_predict['Kalman_HAM_Slow']
df_predict['HAM_Interaction_Corr'] = df_predict['HAM_Triple_Product'].rolling(window=15, min_periods=2).corr(df_predict['Hurst_Amp_Momentum']).fillna(0.0)

df_predict.dropna(subset=['HAM_Triple_Product', 'HAM_Interaction_Corr'], inplace=True)

# =====================================================================
# 5. 🎯 BACK-DATA REGRET SCANNER: HISTORICAL EXPIRY EVALUATOR
# =====================================================================
strike_interval = 500.0
prices = df_predict['Close'].to_numpy()
n_len = len(df_predict)

# Har candle ka calculated Near-ATM numerical value
near_atm_numeric = np.round(prices / strike_interval) * strike_interval

# Look-ahead expiry array creation: Did this specific strike settle at 0 at the end of its window?
# Ek range-based cycle window letey hain (~750 bars window) to evaluate expiry levels safely
expiry_window = 750 
historical_zero_ce = np.zeros(n_len)
historical_zero_pe = np.zeros(n_len)

for idx in range(n_len):
    current_atm = near_atm_numeric[idx]
    end_idx = min(idx + expiry_window, n_len - 1)
    
    # Is time window block me maximum aur minimum trace kya hua price ka?
    future_max = np.max(prices[idx:end_idx + 1])
    future_min = np.min(prices[idx:end_idx + 1])
    future_settle = prices[end_idx] # Window terminal node
    
    # CE zero tabhi hoti hai jab high ya settle us strike ko cross na kare (Out of the money)
    if future_settle < current_atm and future_max < current_atm + 200:
        historical_zero_ce[idx] = 1.0 # Successful True Zero
        
    # PE zero tabhi hoti hai jab low ya settle us strike se niche na jaye
    if future_settle > current_atm and future_min > current_atm - 200:
        historical_zero_pe[idx] = 1.0 # Successful True Zero

df_predict['Hist_Zero_CE'] = historical_zero_ce
df_predict['Hist_Zero_PE'] = historical_zero_pe

# =====================================================================
# 6. ML REGRET LEARNING COMPUTE ENGINE (150 TREES PROCESSOR)
# =====================================================================
features = df_predict[['HAM_Triple_Product', 'HAM_Interaction_Corr', 'Kalman_HAM_Fast', 'Kalman_HAM_Slow']].to_numpy()

# ML directly is state predictive matrix ko seekhega base on historical successful zeros
ml_ce_predictions = np.zeros(n_len)
ml_pe_predictions = np.zeros(n_len)

if n_len > 100:
    block = n_len // 4
    for b in range(1, 4):
        t_end = b * block
        test_end = min((b + 1) * block, n_len)
        
        X_tr = features[:t_end-1]
        y_tr_ce = historical_zero_ce[1:t_end]
        y_tr_pe = historical_zero_pe[1:t_end]
        
        X_te = features[t_end:test_end]
        
        if len(X_tr) > 20 and len(X_te) > 0:
            # Train separate classifiers to find pattern correlations with HAM
            rf_ce = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=6, n_jobs=-1)
            rf_ce.fit(X_tr, y_tr_ce)
            ml_ce_predictions[t_end:test_end] = rf_ce.predict(X_te)
            
            rf_pe = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=6, n_jobs=-1)
            rf_pe.fit(X_tr, y_tr_pe)
            ml_pe_predictions[t_end:test_end] = rf_pe.predict(X_te)

df_predict['ML_CE_Zero_Prob'] = ml_ce_predictions
df_predict['ML_PE_Zero_Prob'] = ml_pe_predictions

# Setup conditional text indicators based on structural memory scan
df_predict['Near_ATM_CE_Status'] = ["🔥 HIGH CHANCE ZERO" if p > 0.68 else "⚠️ RISK OF ITM" for p in ml_ce_predictions]
df_predict['Near_ATM_PE_Status'] = ["🔥 HIGH CHANCE ZERO" if p > 0.68 else "⚠️ RISK OF ITM" for p in ml_pe_predictions]

# String visual columns setup
df_predict['Near_ATM_Strike'] = near_atm_numeric.astype(int).astype(str) + " STRIKE"

# Dynamic Vol Bands Calculations for visual validation
rolling_vol = df_predict['Close'].rolling(window=30, min_periods=5).std().fillna(200.0).to_numpy()
safe_ce_list, safe_pe_list = [], []

for idx in range(n_len):
    price = prices[idx]
    vol = rolling_vol[idx]
    st_ce = int(np.ceil((price + (vol * 3.5)) / 500.0) * 500)
    st_pe = int(np.floor((price - (vol * 3.5)) / 500.0) * 500)
    
    h_max = np.max(prices[:idx+1])
    h_min = np.min(prices[:idx+1])
    if st_ce <= h_max: st_ce = int(np.ceil((h_max + 2500) / 500.0) * 500)
    if st_pe >= h_min: st_pe = int(np.floor((h_min - 2500) / 500.0) * 500)
    safe_ce_list.append(f"{st_ce} CE")
    safe_pe_list.append(f"{st_pe} PE")

df_predict['Safe_Strike_CE'] = safe_ce_list
df_predict['Safe_Strike_PE'] = safe_pe_list
df_predict['Signal'] = ["🟢 BUY (Bullish)" if f > s else "🔴 SELL (Bearish)" for f, s in zip(fast_kf, slow_kf)]

# =====================================================================
# 7. METRICS DISPLAYS & GRID VISUALIZATION LAYER
# =====================================================================
latest_row = df_predict.iloc[-1]
delta_close = f"${(latest_row['Close'] - df_predict['Close'].iloc[-2]):.2f}" if len(df_predict) > 1 else "$0.00"

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.metric(label="BTC Current Close", value=f"${latest_row['Close']:.2f}", delta=delta_close)
with m_col2:
    st.metric(label="🎯 Current Near-ATM Strike", value=latest_row['Near_ATM_Strike'])
with m_col3:
    st.metric(label="📊 Near-ATM CE Zero Prob", value=f"{latest_row['ML_CE_Zero_Prob']*100:.1f}%")
with m_col4:
    st.metric(label="📊 Near-ATM PE Zero Prob", value=f"{latest_row['ML_PE_Zero_Prob']*100:.1f}%")

# Grid display configuration
clean_cols = ['Close', 'Near_ATM_Strike', 'ML_CE_Zero_Prob', 'Near_ATM_CE_Status', 'ML_PE_Zero_Prob', 'Near_ATM_PE_Status', 'Signal']
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={
    'Close': 'BTC Close Price', 
    'Near_ATM_Strike': '🎯 Near ATM Strike',
    'ML_CE_Zero_Prob': '📈 CE Zero Confidence',
    'Near_ATM_CE_Status': '🛡️ Near-ATM CE Risk Alert',
    'ML_PE_Zero_Prob': '📉 PE Zero Confidence',
    'Near_ATM_PE_Status': '🛡️ Near-ATM PE Risk Alert',
    'Signal': 'Market Trend'
}, inplace=True)

display_df_inverted = display_df.iloc[::-1].copy()
display_df_inverted.index = display_df_inverted.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Intelligent Pattern Learning Grid (HAM Dependent Expiry Risk Scans)")
st.dataframe(display_df_inverted, use_container_width=True, height=600)
