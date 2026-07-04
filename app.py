import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="BTC Prob Crossover Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Hour Standalone [Probability Kalman Crossover Engine]")
st.write("🎯 **Aapki Custom Setting:** Strictly Only BTC 1-Hour Data + 50:50 Split + ML Score $[-5,5]$ + Past 25-Candle Target + **🔥 NEW: 25-Candle Kalman Avg on Prob_Up (P=0.50)** + **🔥 NEW: Prob_Up vs Kalman_Avg Crossover Signal** + **Open Score Column REMOVED** + Latest Active Candle Locked on Top")

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

with st.spinner("Calibrating 25-Candle Probability Kalman Filters..."):
    # Bitcoin 1-HOUR Interval Data
    raw_df = yf.download("BTC-USD", period="730d", interval="1h")
    
    if len(raw_df) == 0:
        st.error("YFinance API Timeout. Please refresh the dashboard.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df.index = pd.to_datetime(df.index)

    # Data-Gap Cleaner Filter: Removing zero volume hours
    df = df[df['Volume'] > 0].copy()

    # Base Matrix Definition (Price Kalman 1 Active)
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0, q_val=0.001, r_val=0.1)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Microstructure Features Space
    df['Sign_Change'] = (np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))).astype(int)
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(window=24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Target Rule (Strict Past 25 Candles Core - Zero Leakage)
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    
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
    # 🎯 NEW CORE MATHEMATICAL OPERATION: KALMAN AVG ON PROB_UP (P=0.50)
    # -----------------------------------------------------------------
    # Creating a 25-candle historical lookback shift baseline for probability mapping
    df_predict['Prob_Up_25H_Shift'] = df_predict['Prob_Up'].shift(25).bfill()
    df_predict['Kalman_Prob_Avg'] = apply_kalman_filter_custom(df_predict['Prob_Up_25H_Shift'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # -----------------------------------------------------------------
    # DIRECT CROSSOVER ALERT ENGINE (PROB_UP vs KALMAN_PROB_AVG)
    # -----------------------------------------------------------------
    p_up_arr = df_predict['Prob_Up'].to_numpy()
    k_avg_arr = df_predict['Kalman_Prob_Avg'].to_numpy()
    crossover_log = []

    for idx in range(len(p_up_arr)):
        if p_up_arr[idx] > k_avg_arr[idx]:
            crossover_log.append("🚀 BUY ZONE (Prob > Kalman Avg)")
        elif p_up_arr[idx] < k_avg_arr[idx]:
            crossover_log.append("📉 SELL ZONE (Prob < Kalman Avg)")
        else:
            crossover_log.append("🔷 TIE ZONE")

    df_predict['Crossover_Signal'] = crossover_log

    # Formatting UI Structure (Open Score Removed completely)
    clean_display_cols = ['a_Close', 'b_Kalman_Price', 'Prob_Up', 'Kalman_Prob_Avg', 'Crossover_Signal', 'Accumulator_Score', 'Weighted_Momentum', 'd_ML_Signal']
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Kalman_Prob_Avg'] = display_df['Kalman_Prob_Avg'].round(3)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    
    # Inverting via index flip sequence to lock the absolute latest candle on Top Row
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Probability Kalman Crossover Dashboard (Latest Hour Locked on Top Row)")
    st.dataframe(display_df, use_container_width=True, height=750)
