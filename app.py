import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Momentum Probability Engine", layout="wide")
st.title("⚡ Nifty 50 Live 1-Hour Pure Momentum Probability Engine")
st.write("🎯 **Momentum Core Framework:** Model predicts the directional shift of the Weighted Momentum Wave itself, not just raw price.")

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

with st.spinner("🚀 Mapping Momentum Waves & Aligning Probability Matrix..."):
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
    
    # Calculate Momentum Wave First (Strict Math Layer)
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Generate Weighted Momentum BEFORE Model Training
    df['Weighted_Momentum'] = apply_kalman_filter_custom(df['c_Combined'].values, initial_p=0.50)
    
    # =====================================================================
    # 🆕 THE ULTIMATE SHIFT: Target is now based on Momentum Direction
    # =====================================================================
    df['Target'] = np.where(df['Weighted_Momentum'] > df['Weighted_Momentum'].shift(1), 1, 0)
    
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
    # Train model to predict Momentum Direction
    model_flow = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42)
    model_flow.fit(X_train, y_train)

    # Probabilities of Momentum moving Up or Down
    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # Price Action Columns for confirmation
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

        # Accumulator strictly tracks Momentum Probability Waves
        if p_up >= 0.55: accumulator += 1  
        elif p_down >= 0.55: accumulator -= 1  
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)

        trap_msg = "TREND VALID"

        if accumulator == 5:
            current_state = "BUY"
            if c_val > p_high: final_signals.append("🟢 STRONG MOMENTUM BUY (Max [5/5])")
            else:
                final_signals.append("❌ NO ENTRY (Wait for Breakout)")
                trap_msg = "⚠️ BULL TRAP"
        elif accumulator == -5:
            current_state = "SELL"
            if c_val < p_low: final_signals.append("🔴 STRONG MOMENTUM SELL (Max [-5/-5])")
            else:
                final_signals.append("🟢 HOLD LONG (No Short Entry)")
                trap_msg = "⚠️ BEAR TRAP"
        else:
            if current_state == "BUY":
                if accumulator > 0: final_signals.append(f"🟢 HOLD BUY | Momentum Slowing (Score: {accumulator})")
                else:
                    if c_val < p_low: final_signals.append(f"⚠️ MOMENTUM CRITICAL | Reversal (Score: {accumulator})")
                    else:
                        final_signals.append(f"🔄 HOLD BUY | Fake Wave Dip (Score: {accumulator})")
            elif current_state == "SELL":
                if accumulator < 0: final_signals.append(f"🔴 HOLD SELL | Momentum Building (Score: {accumulator})")
                else:
                    if c_val > p_high: final_signals.append(f"⚠️ MOMENTUM CRITICAL | Reversal (Score: {accumulator})")
                    else:
                        final_signals.append(f"🔄 HOLD SELL | Fake Wave Pump (Score: {accumulator})")
            else:
                final_signals.append(f"⚪ NEUTRAL | Waiting for Momentum Velocity (Score: {accumulator})")

        # Dynamic trap mapping to clean display
        if "TRAP" not in trap_msg:
            trap_msg = "MOMENTUM STABLE" if abs(accumulator) >= 3 else "LOW VELOCITY"

    df_predict['d_ML_Signal'] = final_signals
    df_predict['Trap_Status'] = "TREND VALID" # Placeholder to maintain shape

    # Table View Layout Configuration
    clean_display_cols = [
        'a_Close', 'b_Kalman_Price', 'Prev_High', 'Prev_Low', 
        'Weighted_Momentum', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 
        'd_ML_Signal'
    ]
    
    df_predict['Accumulator_Score'] = scores_log
    display_df = df_predict[clean_display_cols].copy().sort_index(ascending=False)
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Prev_High'] = display_df['Prev_High'].round(2)
    display_df['Prev_Low'] = display_df['Prev_Low'].round(2)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(4)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    display_df['Accumulator_Score'] = display_df['Accumulator_Score'].astype(int)
    
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live 1-Hour Pure Momentum Probability Dashboard")
    st.dataframe(display_df, use_container_width=True, height=750)
