import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Advanced Volatility Engine", layout="wide")
st.title("⚡ Nifty 50 Advanced 25-Candle Quant Engine")
st.write("🎯 **Alpha Core:** Uses Kalman-Filtered True Range & Rolling Volatility Bands to predict institutional expansion.")

# =====================================================================
# MATHEMATICAL ENGINE: LINEAR FILTER
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=100.0): 
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = initial_p  
    q = 0.0001     # Process noise
    r = 2.5        # Measurement noise
    
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("🚀 Booting Quant Engine & Training Alpha Core..."):
    # 1. Download Data
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = [col[0] for col in raw_df.columns]
    
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = pd.to_numeric(raw_df[col], errors='coerce').ffill()
            
    df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)

    # =====================================================================
    # ADVANCED QUANT MATRIX (True Range & Shift Logic)
    # =====================================================================
    df['a_Close'] = df['Close']
    df['Prev_Close'] = df['a_Close'].shift(1).fillna(df['a_Close'])
    
    # ADVANCED: True Range calculation (captures gaps perfectly)
    df['H_L'] = df['High'] - df['Low']
    df['H_PC'] = (df['High'] - df['Prev_Close']).abs()
    df['L_PC'] = (df['Low'] - df['Prev_Close']).abs()
    df['True_Range'] = df[['H_L', 'H_PC', 'L_PC']].max(axis=1)
    
    # Smooth the True Range using Kalman
    df['Volatility_Momentum'] = apply_kalman_filter_custom(df['True_Range'].values, initial_p=0.50)
    
    # DYNAMIC BAND: Calculate 25-candle rolling mean and standard deviation of Volatility
    df['Vol_Mean_25'] = df['Volatility_Momentum'].shift(1).rolling(window=25).mean()
    df['Vol_Std_25'] = df['Volatility_Momentum'].shift(1).rolling(window=25).std()
    df['Vol_Upper_Band'] = df['Vol_Mean_25'] + (1.5 * df['Vol_Std_25']) # 1.5x Threshold
    
    # TARGET RE-DESIGNED: 1 only if Volatility breaks above the 25-candle Upper Band (Real Expansion)
    df['Target'] = np.where(df['Volatility_Momentum'] > df['Vol_Upper_Band'], 1, 0)
    
    # Smart Alpha Features
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=100.0)
    df['Price_Velocity'] = df['a_Close'] - df['b_Kalman_Price']
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['True_Range'] + 1e-10)
    df['RSI_Range'] = df['Volatility_Momentum'].diff(1)
    df['Price_Trend'] = df['a_Close'].diff(5) # 5-candle basic direction
    
    features_matrix = ['Price_Velocity', 'Order_Imbalance', 'RSI_Range', 'Price_Trend']
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features_matrix + ['Target']).copy()

# 50:50 Train-Test Split
split_idx = int(len(df_clean) * 0.50)

df_train = df_clean.iloc[:split_idx].copy()
X_train = df_train[features_matrix]
y_train = df_train['Target']

df_predict = df_clean.iloc[split_idx:].copy()
X_predict = df_predict[features_matrix]

if len(X_predict) > 0 and len(X_train) > 0:
    # Train Regularized Model
    model_flow = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model_flow.fit(X_train, y_train)

    # Predictions
    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Expansion'] = probabilities[:, 1] # Probability of breakdown/breakout burst

    # Trend Indicators
    df_predict['Prev_High'] = df_predict['High'].shift(1).fillna(df_predict['High'])
    df_predict['Prev_Low'] = df_predict['Low'].shift(1).fillna(df_predict['Low'])

    final_signals = []
    closes = df_predict['a_Close'].to_numpy()
    p_expands = df_predict['Prob_Expansion'].to_numpy()
    prev_highs = df_predict['Prev_High'].to_numpy()
    prev_lows = df_predict['Prev_Low'].to_numpy()
    trends = df_predict['Price_Trend'].to_numpy()

    for i in range(len(p_expands)):
        prob = p_expands[i]
        c_val = closes[i]
        p_high = prev_highs[i]
        p_low = prev_lows[i]
        curr_trend = trends[i]

        # FILTER LOGIC: Volatility probability must be high (> 58%) to take a trade
        if prob >= 0.58:
            if curr_trend > 0 and c_val > p_high:
                final_signals.append("🟢 STRONG BREAKOUT BUY")
            elif curr_trend < 0 and c_val < p_low:
                final_signals.append("🔴 STRONG BREAKDOWN SELL")
            else:
                final_signals.append("⚡ VOLATILITY RISING (No Direction)")
        else:
            final_signals.append("💤 CHOPPY / NO TRADING ZONE")

    df_predict['d_ML_Signal'] = final_signals

    # Display Configuration Block
    clean_display_cols = ['a_Close', 'Volatility_Momentum', 'Prob_Expansion', 'd_ML_Signal']
    display_df = df_predict[clean_display_cols].copy().sort_index(ascending=False)
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['Volatility_Momentum'] = display_df['Volatility_Momentum'].round(4)
    display_df['Prob_Expansion'] = (display_df['Prob_Expansion'] * 100).round(1).astype(str) + '%'
    
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader("📋 Live Filtered Quant Trading Dashboard")
    st.dataframe(display_df, use_container_width=True, height=750)
else:
    st.error("Data processing mismatch.")
