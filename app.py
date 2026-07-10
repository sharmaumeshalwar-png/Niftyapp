import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Dual Momentum Engine", layout="wide")
st.title("📊 Nifty 50 Live 1-Hour Hybrid Double Kalman [0.50 Engine]")
st.write("🎯 **Aapki Custom Setting:** Strictly Only Nifty 50 Index Data + Strictly Past 25-Candle Window + Double Filtered Weighted Momentum Layer + Instant Linear & Non-Linear ML Column Arrays")

# =====================================================================
# MATHEMATICAL ENGINE 1: LINEAR FILTER
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=100.0): 
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = initial_p  
    
    q = 0.0001     
    r = 2.5        
    
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

# =====================================================================
# MATHEMATICAL ENGINE 2: NON-LINEAR FILTER
# =====================================================================
def apply_non_linear_kalman_momentum(data_array):
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = 1.0  
    q = 0.05   
    r = 0.2    
    
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("Aligning 25-Candle Dual Kalman Nifty Microstructure Matrices..."):
    raw_df = yf.download("^NSEI", period="1y", interval="1h")
    
    if raw_df.empty:
        raw_df = yf.download("^NSEI", period="1mo", interval="1h")
        
    if raw_df.empty:
        st.error("YFinance API Timeout or Indian Market Closed.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = raw_df[col].ffill()

    df.dropna(inplace=True)
    
    # Base Matrix
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=100.0)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Microstructure Features
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(1), 1, 0)
    df_clean = df.dropna(subset=features_matrix + ['Target']).tail(25).copy()

# =====================================================================
# DYNAMIC SPLIT ENGINE
# =====================================================================
split_idx = int(len(df_clean) * 0.50)
df_train, df_predict = df_clean.iloc[:split_idx], df_clean.iloc[split_idx:]

if len(df_predict) > 0:
    X_train, y_train = df_train[features_matrix], df_train['Target']
    model_flow = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42).fit(X_train, y_train)
    probs = model_flow.predict_proba(df_predict[features_matrix])
    
    df_predict['Prob_Up'] = probs[:, 1]
    df_predict['Prob_Down'] = probs[:, 0]
    
    # Trend Lock & Signals Logic
    accumulator = 0
    final_signals, scores_log, trap_log = [], [], []
    current_state = "HOLD"
    
    for i in range(len(df_predict)):
        p_up, p_down, c_val = df_predict['Prob_Up'].iloc[i], df_predict['Prob_Down'].iloc[i], df_predict['a_Close'].iloc[i]
        p_high, p_low = df_predict['High'].shift(1).iloc[i], df_predict['Low'].shift(1).iloc[i]
        
        if p_up >= 0.55: accumulator += 1
        elif p_down >= 0.55: accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        
        if accumulator == 5:
            current_state = "BUY"
            sig = "🟢 STRONG BUY" if c_val > p_high else "❌ NO ENTRY (Bull Trap)"
        elif accumulator == -5:
            current_state = "SELL"
            sig = "🔴 STRONG SELL" if c_val < p_low else "🟢 HOLD LONG (Bear Trap)"
        else:
            sig = f"🔄 Neutral/Hold (Score: {accumulator})"
            
        final_signals.append(sig)
        scores_log.append(accumulator)
        trap_log.append("TREND VALID" if accumulator in [-5, 5] else "⚠️ TRAP ZONE")

    df_predict['d_ML_Signal'] = final_signals
    df_predict['Accumulator_Score'] = scores_log
    df_predict['Trap_Status'] = trap_log

    st.subheader("📋 Live Nifty Dual Momentum Results")
    st.dataframe(df_predict[['a_Close', 'Prob_Up', 'Accumulator_Score', 'd_ML_Signal', 'Trap_Status']])
else:
    st.error("Insufficient data for split.")
