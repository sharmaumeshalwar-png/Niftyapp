import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="BTC Fixed Spread Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Hour Standalone [Dual Past Kalman Spread Engine - FIXED]")
st.write("🎯 **Aapki Custom Setting:** Strictly Only BTC 1-Hour Data + 50:50 Split + **VWAP completely REMOVED** + ML Score $[-5,5]$ + Strictly Past 10-Candle Target + **Kalman 10-Past & Kalman 25-Past Spread Optimization** + **FIXED: Pandas 2.0+ fillna Deprecation Bug Fixed** + Latest Active Candle Locked on Top")

# =====================================================================
# MATHEMATICAL ENGINE (Flexible Kalman Filter Function)
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

with st.spinner("Processing Fixed Dual Past Kalman Layers & Injecting Spread Matrix..."):
    # Bitcoin 1-HOUR Interval Data (730 Days max for hourly)
    raw_df = yf.download("BTC-USD", period="730d", interval="1h")
    
    if len(raw_df) == 0:
        st.error("YFinance API Timeout. Please refresh the dashboard.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df.index = pd.to_datetime(df.index)

    # Base Matrix Definition (Price Kalman 1 Active)
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0, q_val=0.001, r_val=0.1)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Microstructure Features Space (Note: VWAP features are omitted entirely)
    df['Sign_Change'] = (np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))).astype(int)
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(window=24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # -----------------------------------------------------------------
    # TARGET RULE (STRICT PAST 10 CANDLES - ZERO LEAKAGE)
    # -----------------------------------------------------------------
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(10), 1, 0)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix + ['Target'], inplace=True)

# Dynamic Split Engine (Strict 50:50 Ratio)
split_idx = int(len(df) * 0.50)
df_train = df.iloc[:split_idx]
X_train = df_train[features_matrix].copy()
y_train = df_train['Target'].copy()

df_predict = df.iloc[split_idx:].copy()
X_predict = df_predict[features_matrix].copy()

if len(X_predict) == 0:
    st.error("Prediction matrix error.")
else:
    model_flow = RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=1, random_state=42)
    model_flow.fit(X_train, y_train)

    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # Live Signals & Accumulators
    final_signals, scores_log, raw_weighted_momentum_log = [], [], []
    accumulator = 0
    
    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()

    for i in range(len(prob_ups)):
        p_up, p_down, c_val, k_price_val = prob_ups[i], prob_downs[i], closes[i], kalmans_price[i]
        if p_up >= 0.55: accumulator += 1
        elif p_down >= 0.55: accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)
        raw_weighted_momentum_log.append(c_val - k_price_val)
        
        if accumulator == 5: final_signals.append("🟢 STRONG BUY (Max Locked [5/5])")
        elif accumulator == -5: final_signals.append("🔴 STRONG SELL (Max Locked [-5/-5])")
        else: final_signals.append(f"⚪ NEUTRAL/HOLD (Score: {accumulator})")

    df_predict['d_ML_Signal'] = final_signals
    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 

    # [Kalman 2 Execution] Runs on Raw_Weighted_Momentum
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # -----------------------------------------------------------------
    # 🎯 FIXED CORE CHASSIS: DUAL PAST KALMAN GENERATION (NO DEPRECATION BUG)
    # -----------------------------------------------------------------
    # Enforcing modern .bfill() syntax to bypass line 118/119 crashes smoothly
    df_predict['WM_10_Past_Raw'] = df_predict['Weighted_Momentum'].shift(10).bfill()
    df_predict['WM_25_Past_Raw'] = df_predict['Weighted_Momentum'].shift(25).bfill()

    # Executing 0.50 initial Kalman filters over the past raw arrays
    df_predict['Kalman_10_Past'] = apply_kalman_filter_custom(df_predict['WM_10_Past_Raw'].values, initial_p=0.50, q_val=0.001, r_val=0.1)
    df_predict['Kalman_25_Past'] = apply_kalman_filter_custom(df_predict['WM_25_Past_Raw'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # Deriving the core net tracking spread
    df_predict['K3_Spread_Discovery'] = df_predict['Kalman_10_Past'] - df_predict['Kalman_25_Past']

    # -----------------------------------------------------------------
    # UNBOUNDED EXPANSION ENGINE BASED STRICTLY ON SPREAD VELOCITY
    # -----------------------------------------------------------------
    spread_values = df_predict['K3_Spread_Discovery'].to_numpy()
    k3_unbounded_log = []
    unbounded_accumulator = 0  

    for idx in range(len(spread_values)):
        if idx == 0:
            k3_unbounded_log.append(0)
            continue
        if spread_values[idx] > spread_values[idx - 1]: unbounded_accumulator += 1
        elif spread_values[idx] < spread_values[idx - 1]: unbounded_accumulator -= 1
        k3_unbounded_log.append(unbounded_accumulator)
        
    df_predict['K3_Open_Score'] = k3_unbounded_log

    # Formatting UI Structure
    clean_display_cols = ['a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Weighted_Momentum', 'Kalman_10_Past', 'Kalman_25_Past', 'K3_Spread_Discovery', 'K3_Open_Score', 'd_ML_Signal']
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    display_df['Kalman_10_Past'] = display_df['Kalman_10_Past'].round(2) 
    display_df['Kalman_25_Past'] = display_df['Kalman_25_Past'].round(2) 
    display_df['K3_Spread_Discovery'] = display_df['K3_Spread_Discovery'].round(2) 
    display_df['K3_Open_Score'] = display_df['K3_Open_Score'].astype(int)
    
    # Inverting completely via index flip to force the live active bar on Top Row
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live 1-Hour Dual Past Kalman Spread Matrix (Latest Active Hour Locked on Top Row)")
    st.dataframe(display_df, use_container_width=True, height=750)
