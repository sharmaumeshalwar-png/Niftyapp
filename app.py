import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Volatility Range Engine", layout="wide")
st.title("⚡ Nifty 50 Live 1-Hour High-Low Volatility Range Engine")
st.write("🎯 **Volatility Squeeze Framework:** Model strictly predicts the expansion (Breakout Power) or contraction (Squeeze) of the High-Low candle range.")

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

with st.spinner("🚀 Scanning Intraday Spreads & Training Volatility Engine..."):
    # Safe single-layer column download
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
    
    # 🆕 CORE VOLATILITY MATRIX: High - Low (Raw Range)
    df['Candle_Range'] = df['High'] - df['Low']
    
    # Structural Features Matrix
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['Candle_Range'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['Candle_Range'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['Candle_Range'].diff(1) 
    
    # 🎯 TARGET CHANGE: Predict if Raw Range expands vs 1 Candle Past
    df['Target'] = np.where(df['Candle_Range'] > df['Candle_Range'].shift(1), 1, 0)
    
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
    # Original Trusted Tree Structure
    model_flow = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42)
    model_flow.fit(X_train, y_train)

    # Probabilities Generation (Up = Range Expanding, Down = Range Squeezing)
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

        # Accumulator strictly tracks Volatility Expansion State
        if p_up >= 0.55: accumulator += 1  
        elif p_down >= 0.55: accumulator -= 1  
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)

        if accumulator == 5:
            current_state = "EXPANSION"
            if c_val > p_high: final_signals.append("🟢 BREAKOUT BUY (High Volatility Expansion)")
            elif c_val < p_low: final_signals.append("🔴 BREAKDOWN SELL (High Volatility Expansion)")
            else: final_signals.append("⚡ VOLATILITY ALERT (Range Broadening)")
        elif accumulator == -5:
            current_state = "SQUEEZE"
            final_signals.append("⚪ CRITICAL SQUEEZE (Market Freezing / No Entry)")
        else:
            if current_state == "EXPANSION":
                if accumulator > 0: final_signals.append(f"🔄 HOLD POSITION | Volatility Easing ({accumulator})")
                else: final_signals.append(f"⚠️ TREND FADE | Volatility Dropped ({accumulator})")
            else:
                final_signals.append(f"💤 SIDEWAYS | Squeeze Continuing ({accumulator})")

    df_predict['d_ML_Signal'] = final_signals
    df_predict['Accumulator_Score'] = scores_log

    # Table View Layout Configuration
    clean_display_cols = [
        'a_Close', 'Candle_Range', 'Prob_Up', 'Prob_Down', 
        'Accumulator_Score', 'd_ML_Signal'
    ]
    
    display_df = df_predict[clean_display_cols].copy().sort_index(ascending=False)
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['Candle_Range'] = display_df['Candle_Range'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    display_df['Accumulator_Score'] = display_df['Accumulator_Score'].astype(int)
    
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live 1-Hour Nifty High-Low Volatility Dashboard")
    st.dataframe(display_df, use_container_width=True, height=750)
