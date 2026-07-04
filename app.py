import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="BTC Past 10H No-Leakage Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Hour Standalone [Strict Past 10-Hour Momentum Engine]")
st.write("🎯 **Aapki Custom Setting:** Strictly Only BTC 1-Hour Data + 50:50 Split + Velocity Useful_VWAP + ML Score $[-5,5]$ + Unbounded Open Expansion on Kalman 2 + **🔥 NEW FIXED: Strictly Past 10-Candle Momentum (Zero Future Leakage)** + Latest Active Candle Locked on Top")

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

with st.spinner("Calibrating Engine for Fast 10-Hour Past Momentum Waves..."):
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
    
    # Intraday Base VWAP
    typical_price = (df['High'] + df['Low'] + df['a_Close']) / 3
    df['TP_Vol'] = typical_price * df['Volume']
    df['Date_Group'] = df.index.date
    df['VWAP'] = df.groupby('Date_Group')['TP_Vol'].cumsum() / (df.groupby('Date_Group')['Volume'].cumsum() + 1e-10)
    
    # Microstructure Features Space
    df['Sign_Change'] = (np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))).astype(int)
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(window=24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # -----------------------------------------------------------------
    # 🎯 STRICT PAST NO-LEAKAGE TARGET RULE (10 CANDLES):
    # shift(10) looks back at the PAST 10 candles.
    # -----------------------------------------------------------------
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(10), 1, 0)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix + ['Target', 'VWAP'], inplace=True)

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

    # Kalman 2 Execution
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # Useful VWAP Velocity
    df_predict['Useful_VWAP'] = df_predict['VWAP'] * (df_predict['Prob_Up'].shift(1) - df_predict['Prob_Up'])
    df_predict['Useful_VWAP'] = df_predict['Useful_VWAP'].fillna(0)

    # Unbounded Open Score Logic Direct on Kalman 2
    k2_values = df_predict['Weighted_Momentum'].to_numpy()
    k2_unbounded_log = []
    unbounded_accumulator = 0  

    for idx in range(len(k2_values)):
        if idx == 0:
            k2_unbounded_log.append(0)
            continue
        if k2_values[idx] > k2_values[idx - 1]: unbounded_accumulator += 1
        elif k2_values[idx] < k2_values[idx - 1]: unbounded_accumulator -= 1
        k2_unbounded_log.append(unbounded_accumulator)
        
    df_predict['K2_Open_Score'] = k2_unbounded_log

    # Formatting UI Structure
    clean_display_cols = ['a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Weighted_Momentum', 'K2_Open_Score', 'd_ML_Signal']
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    display_df['K2_Open_Score'] = display_df['K2_Open_Score'].astype(int)
    
    # Strictly flip index rows to freeze latest active hour on Top
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live Strict Past 10H Momentum Matrix (Latest Hour Locked on Top Row)")
    st.dataframe(display_df, use_container_width=True, height=750)
