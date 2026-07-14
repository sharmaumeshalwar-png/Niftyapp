import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="BTC Live Backtester Engine", layout="wide")
st.title("⚡ BTC-USD 2-Year Hourly Backtester & Hedging Engine")
st.write("🎯 **Pure Backtest Mode:** Calculates exact win rates, losses, and net dollar earnings based on your 1:3 Ratio Hedge Strategy.")

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

# -----------------------------------------------------------------
# 🛡️ DIRECT REAL TIME DATA ONLY (2-YEAR HIGH DENSITY CRYPTO STREAM)
# -----------------------------------------------------------------
df = None
selected_period = "2y"   
selected_interval = "1h" 

with st.spinner("Fetching 2-Year Hourly Live BTC Data directly from Exchange Server..."):
    try:
        df = yf.download(tickers="BTC-USD", period=selected_period, interval=selected_interval)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    except Exception as e:
        st.error(f"🚨 API Connection Failed: {e}")
        st.stop()

st.success(f"🟢 Successfully Synced {len(df)} Candles!")

# Base Matrix Definition
df['a_Close'] = df['Close']
df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, q_val=0.001, r_val=0.1)
df['Slow_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, q_val=0.00001, r_val=0.9)
df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']

# TUNNEL CALCULATIONS
wma_weights = np.arange(12, 0, -1) 
wma_sum = np.sum(wma_weights)       
df['Fast_WMA_Tunnel'] = df['b_Kalman_Price'].rolling(window=12).apply(lambda x: np.sum(x * wma_weights) / wma_sum, raw=True)
df['Slow_WMA_Tunnel'] = df['Slow_Kalman_Price'].rolling(window=12).apply(lambda x: np.sum(x * wma_weights) / wma_sum, raw=True)

df['Target_Next_Direction'] = np.where(df['a_Close'].shift(-1) > df['a_Close'], 1, 0)
df.ffill().bfill()

# Dynamic Split Engine (50:50)
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

# Probabilities
probabilities = model_flow.predict_proba(df_predict[gravity_features])
df_predict['Prob_Up_Raw'] = probabilities[:, 1]
df_predict['Prob_Down_Raw'] = probabilities[:, 0]

# Signals
price_vals = df_predict['a_Close'].to_numpy()
fast_vals = df_predict['b_Kalman_Price'].to_numpy()
slow_vals = df_predict['Slow_Kalman_Price'].to_numpy()
fast_wma = df_predict['Fast_WMA_Tunnel'].to_numpy()
slow_wma = df_predict['Slow_WMA_Tunnel'].to_numpy()

signal_log = []
for idx in range(len(fast_vals)):
    if np.isnan(fast_wma[idx]) or np.isnan(slow_wma[idx]):
        signal_log.append("⏳ LOADING")
        continue
    fast_bullish = (fast_vals[idx] > fast_wma[idx]) and (price_vals[idx] > fast_vals[idx])
    slow_bullish = (slow_vals[idx] > slow_wma[idx]) and (price_vals[idx] > slow_vals[idx])
    fast_bearish = (fast_vals[idx] < fast_wma[idx]) and (price_vals[idx] < fast_vals[idx])
    slow_bearish = (slow_vals[idx] < slow_wma[idx]) and (price_vals[idx] < slow_vals[idx])
    
    if fast_bullish and slow_bullish: signal_log.append("🟢 BUY")
    elif fast_bearish and slow_bearish: signal_log.append("🔴 SELL")
    else: signal_log.append("⏳ WAIT ZONE")
df_predict['Signal'] = signal_log

# =====================================================================
# 📈 BACKTEST SIMULATOR ENGINE (1:3 Ratio Backspread Core)
# =====================================================================
net_profit = 0.0
total_trades = 0
reversal_wins = 0
trend_jackpots = 0
flat_losses = 0

# Hum look-ahead use karenge to check trade outcome over next 24 Hours (Expiry window)
look_ahead_candles = 24 

for idx in range(len(df_predict) - look_ahead_candles):
    sig = df_predict['Signal'].iloc[idx]
    entry_price = df_predict['a_Close'].iloc[idx]
    future_price = df_predict['a_Close'].iloc[idx + look_ahead_candles]
    
    # Core variables mapping
    atm_credit = 3000.0   # Average premium collected from ATM sella
    otm_cost = 600.0      # Average premium bought for 1 OTM hedge leg (Total 3x buy = 1800)
    net_entry_credit = atm_credit - (3 * otm_cost) # +1200 guaranteed credit
    
    if sig == "🟢 BUY":
        # Opposite Trade: Bearish (View is market drops)
        total_trades += 1
        price_diff = future_price - entry_price
        
        if price_diff <= 0:
            # Reversal Successful! Market dropped or stayed flat
            reversal_wins += 1
            net_profit += net_entry_credit
        else:
            # Market instead trends up (Trap Zone)
            # OTM Calls explode, offsetting ATM loss
            option_payoff = (3 * max(0, price_diff - (entry_price * 0.05))) - max(0, price_diff)
            trade_result = net_entry_credit + option_payoff
            if trade_result > 0:
                trend_jackpots += 1
            else:
                flat_losses += 1
            net_profit += trade_result
            
    elif sig == "🔴 SELL":
        # Opposite Trade: Bullish (View is market rises)
        total_trades += 1
        price_diff = entry_price - future_price
        
        if price_diff <= 0:
            # Reversal Successful! Market rose or stayed flat
            reversal_wins += 1
            net_profit += net_entry_credit
        else:
            # Market instead trends down (Trap Zone)
            # OTM Puts explode
            option_payoff = (3 * max(0, price_diff - (entry_price * 0.05))) - max(0, price_diff)
            trade_result = net_entry_credit + option_payoff
            if trade_result > 0:
                trend_jackpots += 1
            else:
                flat_losses += 1
            net_profit += trade_result

# Displaying Backtest Results
st.subheader("📊 Real Backtest Outcome Stats")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trades Triggered", f"{total_trades}")
col2.metric("Reversal Wins (70% Class)", f"{reversal_wins}", "Reversal hit")
col3.metric("Trend Jackpot Wins (30% Class)", f"{trend_jackpots}", "Gamma blast")
col4.metric("Net Strategy Profit ($)", f"${net_profit:,.2f}")

# Historical Data Frame inverted for view
df_predict['Net_Accumulated_PnL'] = net_profit # tracking variable
display_df = df_predict[['a_Close', 'Signal', 'Prob_Up_Raw', 'Prob_Down_Raw']].copy().iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Dynamic Pricing and Probability Matrix")
st.dataframe(display_df, use_container_width=True, height=500)
