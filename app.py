import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Master Non-Linear Engine", layout="wide")
st.title("🌀 Nifty 50 Live 1-Hour Master Non-Linear Chaos [0.50 Engine]")
st.write("🎯 **Aapki Custom Master Setting:** 2 Years Data (50:50 Split) + 1-Hour Candles + Past 25-Candle Window + Accumulator + Weighted Momentum Layer + High/Low Trap Filter")

# =====================================================================
# MATHEMATICAL ENGINE (Non-Linear Sigmoid Filter Function)
# =====================================================================
def apply_non_linear_kalman(data_array, initial_p=50.0):
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = initial_p  
    q = 0.005      # Process noise for non-linear spikes
    r = 0.05       # Measurement noise
    filtered_values = []
    
    for z in data_array:
        p = p + q
        k = p / (p + r)
        innovation = z - x
        # Non-Linear Sigmoid Mapping Layer
        non_linear_scale = 2 / (1 + np.exp(-0.01 * innovation)) - 1
        x = x + k * (innovation * (1 + np.abs(non_linear_scale)))
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("Aligning Master Microstructure Matrices (2 Years Chronological Window)..."):
    # 1. 1-Hour Candlestick + 2 Years Historical Data
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    
    if len(raw_df) == 0:
        st.error("API Timeout or Indian Market Closed. Please refresh.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df.index = pd.to_datetime(df.index)

    # Base Matrix Definition
    df['a_Close'] = df['Close']
    df['b_NonLinear_Price'] = apply_non_linear_kalman(df['a_Close'].values, initial_p=50.0)
    df['c_Combined'] = df['a_Close'] - df['b_NonLinear_Price']
    
    # Non-Linear Volatility Squashing & Structural Features
    df['Log_Return'] = np.log(df['a_Close'] / df['a_Close'].shift(1))
    df['NonLinear_Volatility'] = df['Log_Return'].rolling(window=24).std() * df['a_Close']
    df['Normalized_Gap'] = df['c_Combined'] / (df['NonLinear_Volatility'] + 1e-10)
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Normalized_Gap', 'Flow_Velocity']

    # 🎯 STRICT PAST 25-CANDLE TARGET WINDOW
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)

# =====================================================================
# DYNAMIC SPLIT ENGINE (Strict 50:50 Ratio)
# =====================================================================
split_idx = int(len(df) * 0.50)

# 1. Training Set (Pehele 50%)
df_train = df.iloc[:split_idx].copy()
df_train.dropna(subset=features_matrix + ['Target'], inplace=True)

X_train = df_train[features_matrix]
y_train = df_train['Target']

# 2. Prediction Set (Baad ke 50%)
df_predict = df.iloc[split_idx:].copy()
df_predict.dropna(subset=features_matrix, inplace=True) 

X_predict = df_predict[features_matrix]

if len(X_predict) != 0:
    # Gradient Boosting Training for Non-Linear Topologies
    model_flow = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model_flow.fit(X_train, y_train)

    # Live Probabilities Generation (Prob_Up and Prob_Down)
    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # Price action capture (Previous Candle High/Low)
    df_predict['Prev_High'] = df_predict['High'].shift(1)
    df_predict['Prev_Low'] = df_predict['Low'].shift(1)

    # =====================================================================
    # LIVE TREND-LOCK CIRCUIT WITH ACCUMULATOR & TRAP DETECTION
    # =====================================================================
    final_signals = []
    scores_log = []
    raw_weighted_momentum_log = [] 
    trap_status_log = []
    current_state = "HOLD"
    accumulator = 0

    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_NonLinear_Price'].to_numpy()
    prev_highs = df_predict['Prev_High'].to_numpy()
    prev_lows = df_predict['Prev_Low'].to_numpy()

    for i in range(len(prob_ups)):
        p_up = prob_ups[i]
        p_down = prob_downs[i]
        c_val = closes[i]
        k_price_val = kalmans_price[i]
        p_high = prev_highs[i] if not np.isnan(prev_highs[i]) else c_val
        p_low = prev_lows[i] if not np.isnan(prev_lows[i]) else c_val

        # --- 1. ACCUMULATOR ENGINE COUNTER ---
        if p_up >= 0.55:
            accumulator += 1
        elif p_down >= 0.55:
            accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)

        # --- 2. RAW WEIGHTED MOMENTUM VECTOR ---
        calc_raw_weighted = c_val - k_price_val
        raw_weighted_momentum_log.append(calc_raw_weighted)

        # --- 3. PRICE ACTION CONFIRMATION CIRCUIT ---
        trap_msg = "TREND VALID"

        if accumulator == 5:
            current_state = "BUY"
            if c_val > p_high: final_signals.append("🟢 NON-LINEAR STRONG BUY")
            else:
                final_signals.append("❌ NO ENTRY (Wait for High Break)")
                trap_msg = "⚠️ BULL TRAP (High Not Broken)"
        elif accumulator == -5:
            current_state = "SELL"
            if c_val < p_low: final_signals.append("🔴 NON-LINEAR STRONG SELL")
            else:
                final_signals.append("🟢 HOLD LONG (No Short Entry)")
                trap_msg = "⚠️ BEAR TRAP (Low Not Broken)"
        else:
            if current_state == "BUY":
                if accumulator > 0: final_signals.append(f"🟢 HOLD BUY (Score: {accumulator})")
                else:
                    if c_val < p_low: final_signals.append(f"⚠️ REVERSAL CRITICAL (Score: {accumulator})")
                    else:
                        final_signals.append(f"🔄 FAKE DIP (Score: {accumulator})")
                        trap_msg = "⚠️ BEAR TRAP INSIDE TREND"
            elif current_state == "SELL":
                if accumulator < 0: final_signals.append(f"🔴 HOLD SELL (Score: {accumulator})")
                else:
                    if c_val > p_high: final_signals.append(f"⚠️ REVERSAL CRITICAL (Score: {accumulator})")
                    else:
                        final_signals.append(f"🔄 FAKE PUMP (Score: {accumulator})")
                        trap_msg = "⚠️ BULL TRAP INSIDE TREND"
            else:
                final_signals.append(f"⚪ NEUTRAL (Score: {accumulator})")

        trap_status_log.append(trap_msg)

    # Mapping variables to dataframe layout
    df_predict['d_ML_Signal'] = final_signals
    df_predict['Trap_Status'] = trap_status_log
    df_predict['Accumulator_Score'] = scores_log
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log

    # --- 4. DOUBLE KALMAN FILTER FOR WEIGHTED MOMENTUM ---
    df_predict['Weighted_Momentum'] = apply_non_linear_kalman(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50)

    # Display Configuration Layout
    clean_display_cols = [
        'a_Close', 'b_NonLinear_Price', 'Prev_High', 'Prev_Low', 
        'Prob_Up', 'Prob_Down', 'Accumulator_Score', 
        'Weighted_Momentum', 'd_ML_Signal', 'Trap_Status'
    ]
    display_df = df_predict[clean_display_cols].copy().sort_index(ascending=False)
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_NonLinear_Price'] = display_df['b_NonLinear_Price'].round(2)
    display_df['Prev_High'] = display_df['Prev_High'].round(2)
    display_df['Prev_Low'] = display_df['Prev_Low'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    display_df['Accumulator_Score'] = display_df['Accumulator_Score'].astype(int)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live Master Non-Linear Matrix Output")
    st.dataframe(display_df, use_container_width=True, height=750)
