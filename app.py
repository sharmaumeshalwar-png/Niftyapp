import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty 25-Candle Volatility Engine", layout="wide")
st.title("⚡ Nifty 50 Live 1-Hour 25-Candle Weighted Volatility Engine")
st.write("🎯 **Macro Quant Core:** Predicts the structural expansion/contraction of the Volatility Wave compared strictly against **25 Candles Past High-Low Range**.")

# =====================================================================
# MATHEMATICAL ENGINE: LINEAR FILTER (Price Mapping Layer)
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

with st.spinner("🚀 Mapping 25-Candle Volatility Waves & Training Macro Core..."):
    # Safe single-layer column download from yfinance
    raw_df = yf.download("^NSEI", period="2y", interval="1h", multi_level_index=False)
    
    if raw_df.empty:
        raw_df = yf.download("^NSEI", period="1mo", interval="1h", multi_level_index=False)
        
    if raw_df.empty:
        st.error("YFinance API Timeout or Indian Market Closed. Please refresh the dashboard.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].ffill()

    df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)
    df.index = pd.to_datetime(df.index)

    # Base Matrix Definition
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=100.0)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']  
    
    # Raw Range to Weighted Volatility Momentum Wave
    df['Raw_Range'] = df['High'] - df['Low']
    df['Volatility_Momentum'] = apply_kalman_filter_custom(df['Raw_Range'].values, initial_p=0.50)
    
    # Structural Features Matrix
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['Raw_Range'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['Raw_Range'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['Volatility_Momentum'].diff(1) 
    
    # 🎯 TARGET DESIGN CHANGED: High-Low Range matrix compared strictly against 25 Candles Past (t-25)
    df['Target'] = np.where(df['Volatility_Momentum'] > df['Volatility_Momentum'].shift(25), 1, 0)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df_clean = df.replace([np.inf, -np.inf], np.nan).copy()

# Dynamic Split Engine (50:50 Split)
split_idx = int(len(df_clean) * 0.50)

df_train = df_clean.iloc[:split_idx].copy()
df_train.dropna(subset=features_matrix + ['Target'], inplace=True)

X_train = df_train[features_matrix]
y_train = df_train['Target']

df_predict = df_clean.iloc[split_idx:].copy()
df_predict.dropna(subset=features_matrix, inplace=True) 

X_predict = df_predict[features_matrix]

if len(X_predict) == 0 or len(X_train) == 0:
    st.error(f"⚠️ Data size insufficient for split.")
else:
    # Original Trusted Tree Structure (150 estimators, depth 4)
    model_flow = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42)
    model_flow.fit(X_train, y_train)

    # Probabilities Generation
    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # Price Action Columns
    df_predict['Prev_High'] = df_predict['High'].shift(1)
    df_predict['Prev_Low'] = df_predict['Low'].shift(1)

    # Live Circuit Log Generation
    final_signals = []
    scores_log = []
    current_state = "HOLD"
    accumulator = 0

    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    prev_highs = df_predict['Prev_High'].to_numpy()
    prev_lows = df_predict['Prev_Low'].to_numpy()

    for i in range(len(prob_ups)):
        p_up = prob_ups[i]
        p_down = prob_downs[i]
        c_val = closes[i]
        p_high = prev_highs[i] if not np.isnan(prev_highs[i]) else c_val
        p_low = prev_lows[i] if not np.isnan(prev_lows[i]) else c_val

        if p_up >= 0.55: 
            accumulator += 1  
        elif p_down >= 0.55: 
            accumulator -= 1  
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)

        if accumulator == 5:
            current_state = "EXPANSION"
            if c_val > p_high: 
                final_signals.append("🟢 MACRO MOMENTUM BUY (Confirmed Up)")
            elif c_val < p_low: 
                final_signals.append("🔴 MACRO MOMENTUM SELL (Confirmed Down)")
            else: 
                final_signals.append("⚡ WAVE EXPANDING (No Price Breakout)")
        elif accumulator == -5:
            current_state = "SQUEEZE"
            final_signals.append("⚪ COMPRESSION BLOCK (Squeeze Active / No Trade)")
        else:
            if current_state == "EXPANSION":
                if accumulator > 0: 
                    final_signals.append(f"🔄 HOLD POSITION | Macro Wave Cooling ({accumulator})")
                else: 
                    final_signals.append(f"⚠️ TREND FADE | Volatility Dropping ({accumulator})")
            else:
                final_signals.append(f"💤 CHOPPY WAVE | Macro Squeeze ({accumulator})")

    df_predict['d_ML_Signal'] = final_signals
    df_predict['Accumulator_Score'] = scores_log

    # Display Configuration Block
