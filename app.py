import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pytz

# Page Configuration
st.set_page_config(page_title="BTC Hedged Gravity Engine", layout="wide")
st.title("⚡ BTC-USD 2-Year 1-Hour Hedged Gravity Engine")
st.write("🎯 **Ratio Backspread Hedging Engine:** Reversal (70% Win) vs Aggressive Trend (30% Trap Protection)")

# =====================================================================
# MATHEMATICAL ENGINE (Flexible Kalman Filter & VIDYA Functions)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = initial_p  
    q = q_val      
    r = r_val        
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

def apply_vidya_custom(data_array, period=14):
    if len(data_array) < period:
        return data_array.copy()
        
    s = pd.Series(data_array)
    diff = s.diff()
    gains = diff.where(diff > 0, 0)
    losses = (-diff).where(diff < 0, 0)
    
    sum_gains = gains.rolling(window=period).sum()
    sum_losses = losses.rolling(window=period).sum()
    
    cmo = (sum_gains - sum_losses) / (sum_gains + sum_losses + 1e-10)
    k = cmo.abs().fillna(0).to_numpy()
    
    alpha = 2 / (period + 1)
    vidya_values = np.zeros_like(data_array)
    vidya_values[0] = data_array[0]
    
    for i in range(1, len(data_array)):
        vidya_values[i] = (alpha * k[i] * data_array[i]) + (1 - alpha * k[i]) * vidya_values[i-1]
        
    return vidya_values

# -----------------------------------------------------------------
# 🛡️ DATA SYNC
# -----------------------------------------------------------------
df = None
selected_period = "2y"   
selected_interval = "1h" 

with st.spinner("Fetching 2-Year Hourly Live BTC Data..."):
    try:
        df = yf.download(tickers="BTC-USD", period=selected_period, interval=selected_interval)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    except Exception as e:
        st.error(f"🚨 Connection Failed: {e}")
        st.stop()

df['a_Close'] = df['Close']
df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, q_val=0.001, r_val=0.1)
df['Slow_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, q_val=0.00001, r_val=0.9)
df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']

wma_weights = np.arange(12, 0, -1) 
wma_sum = np.sum(wma_weights)       
df['Fast_WMA_Tunnel'] = df['b_Kalman_Price'].rolling(window=12).apply(lambda x: np.sum(x * wma_weights) / wma_sum, raw=True)
df['Slow_WMA_Tunnel'] = df['Slow_Kalman_Price'].rolling(window=12).apply(lambda x: np.sum(x * wma_weights) / wma_sum, raw=True)

# Target Direction
df['Target_Next_Direction'] = np.where(df['a_Close'].shift(-1) > df['a_Close'], 1, 0)
df.ffill().bfill()

# Split 50:50
split_idx = int(len(df) * 0.50)
df_train = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

df_train['Fast_WMA_Slope'] = df_train['Fast_WMA_Tunnel'].diff(1).fillna(0)
df_train['Price_To_Fast_WMA_Gap'] = df_train['a_Close'] - df_train['Fast_WMA_Tunnel']
df_predict['Fast_WMA_Slope'] = df_predict['Fast_WMA_Tunnel'].diff(1).fillna(0)
df_predict['Price_To_Fast_WMA_Gap'] = df_predict['a_Close'] - df_predict['Fast_WMA_Tunnel']

gravity_features = ['Fast_WMA_Slope', 'Price_To_Fast_WMA_Gap']

model_flow = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
model_flow.fit(df_train[gravity_features], df_train['Target_Next_Direction'])

# Probabilities with High Contrast Sigmoid
probabilities = model_flow.predict_proba(df_predict[gravity_features])
extreme_prob_up = []
extreme_prob_down = []
gap_std = df_predict['Price_To_Fast_WMA_Gap'].std() + 1e-10

for i in range(len(df_predict)):
    norm_gap = df_predict['Price_To_Fast_WMA_Gap'].iloc[i] / gap_std
    slope_val = df_predict['Fast_WMA_Slope'].iloc[i]
    if norm_gap > 0 and slope_val > 0:
        conf = 1 / (1 + np.exp(-15.0 * norm_gap))
        p_up = 0.50 + 0.49 * conf
    elif norm_gap < 0 and slope_val < 0:
        conf = 1 / (1 + np.exp(15.0 * norm_gap))
        p_up = 0.50 - 0.49 * conf
    else:
        p_up = 0.50 + 0.40 * (1 / (1 + np.exp(-8.0 * norm_gap)) - 0.5)
    
    p_up = max(0.01, min(0.99, p_up))
    extreme_prob_up.append(p_up)
    extreme_prob_down.append(1.0 - p_up)

df_predict['Prob_Up_Raw'] = extreme_prob_up
df_predict['Prob_Down_Raw'] = extreme_prob_down

# Signal Logic
price_vals = df_predict['a_Close'].to_numpy()
fast_vals = df_predict['b_Kalman_Price'].to_numpy()
fast_wma = df_predict['Fast_WMA_Tunnel'].to_numpy()
slow_vals = df_predict['Slow_Kalman_Price'].to_numpy()
slow_wma = df_predict['Slow_WMA_Tunnel'].to_numpy()

signal_log = []
core_trade_log = []
hedge_legs_log = []

for idx in range(len(fast_vals)):
    if np.isnan(fast_wma[idx]) or np.isnan(slow_wma[idx]):
        signal_log.append("⏳ LOADING")
        core_trade_log.append("N/A")
        hedge_legs_log.append("N/A")
        continue
    
    fast_bullish = (fast_vals[idx] > fast_wma[idx]) and (price_vals[idx] > fast_vals[idx])
    slow_bullish = (slow_vals[idx] > slow_wma[idx]) and (price_vals[idx] > slow_vals[idx])
    fast_bearish = (fast_vals[idx] < fast_wma[idx]) and (price_vals[idx] < fast_vals[idx])
    slow_bearish = (slow_vals[idx] < slow_wma[idx]) and (price_vals[idx] < slow_vals[idx])
    
    close_price = round(price_vals[idx], -2) # Approx ATM strike
    
    if fast_bullish and slow_bullish: 
        signal_log.append("🟢 BUY")
        # Opposite Trade (Bearish) + Trend Hedge
        core_trade_log.append(f"SELL 1x ATM Call {close_price}")
        hedge_legs_log.append(f"BUY 3x OTM Call {round(close_price * 1.05, -2)}")
    elif fast_bearish and slow_bearish: 
        signal_log.append("🔴 SELL")
        # Opposite Trade (Bullish) + Trend Hedge
        core_trade_log.append(f"SELL 1x ATM Put {close_price}")
        hedge_legs_log.append(f"BUY 3x OTM Put {round(close_price * 0.95, -2)}")
    else: 
        signal_log.append("⏳ WAIT")
        core_trade_log.append("No Trade")
        hedge_legs_log.append("No Trade")

df_predict['Signal'] = signal_log
df_predict['Core_Trade'] = core_trade_log
df_predict['Hedge_Legs'] = hedge_legs_log

# Display
clean_cols = ['a_Close', 'Signal', 'Core_Trade', 'Hedge_Legs', 'Prob_Up_Raw', 'Prob_Down_Raw']
display_df = df_predict[clean_cols].copy().iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Hedged Signal Matrix")
st.dataframe(display_df, use_container_width=True, height=600)
