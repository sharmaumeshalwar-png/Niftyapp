import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Nifty Dual Momentum Engine", layout="wide")
st.title("📊 Nifty 50 Live 1-Hour Hybrid Double Kalman [0.50 Engine]")
st.write("🎯 **Strictly Only Nifty 50 Index Data + Strictly Past 25-Candle Window**")

def apply_kalman_filter_custom(data_array, initial_p=100.0): 
    if len(data_array) == 0: return []
    x, p = data_array[0], initial_p  
    q, r = 0.0001, 2.5        
    filtered_values = []
    for z in data_array:
        p += q
        k = p / (p + r)
        x += k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

def apply_non_linear_kalman_momentum(data_array):
    if len(data_array) == 0: return []
    x, p = data_array[0], 1.0  
    q, r = 0.05, 0.2    
    filtered_values = []
    for z in data_array:
        p += q
        k = p / (p + r)
        x += k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("Loading Matrices..."):
    raw_df = yf.download("^NSEI", period="1y", interval="1h")
    if raw_df.empty:
        st.error("Data fetch failed. Refresh target.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].ffill()

    df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)
    
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(1), 1, 0)
    
    df_clean = df.dropna(subset=features_matrix + ['Target']).copy()
    if len(df_clean) >= 25:
        df_clean = df_clean.tail(25)

split_idx = int(len(df_clean) * 0.50)
df_train = df_clean.iloc[:split_idx].copy()
df_predict = df_clean.iloc[split_idx:].copy()

X_train, y_train = df_train[features_matrix], df_train['Target']
X_predict = df_predict[features_matrix]

# 🎯 DYNAMIC PROTECTION LAYER (NO PREDICT_PROBA, NO SAMPLES CRASH)
df_predict['Prob_Up'] = 0.50
df_predict['Prob_Down'] = 0.50

if len(np.unique(y_train)) > 1 and len(df_predict) > 0:
    model_flow = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model_flow.fit(X_train, y_train)
    
    # Direct 1D Hard Predictions mapping (No multi-dim array slicing to hit index error)
    safe_vector_predictions = model_flow.predict(X_predict).astype(float)
    df_predict['Prob_Up'] = safe_vector_predictions
    df_predict['Prob_Down'] = 1.0 - safe_vector_predictions

df_predict['Prev_High'] = df_predict['High'].shift(1)
df_predict['Prev_Low'] = df_predict['Low'].shift(1)

# Circuit Core Accumulator Loop
final_signals, scores_log, raw_wm, trap_log = [], [], [], []
accumulator = 0

for i in range(len(df_predict)):
    row = df_predict.iloc[i]
    p_up, p_down, c_val, k_val = row['Prob_Up'], row['Prob_Down'], row['a_Close'], row['b_Kalman_Price']
    p_high = row['Prev_High'] if not pd.isna(row['Prev_High']) else c_val
    p_low = row['Prev_Low'] if not pd.isna(row['Prev_Low']) else c_val
    
    if p_up >= 0.50: accumulator = min(5, accumulator + 1)
    elif p_down >= 0.50: accumulator = max(-5, accumulator - 1)
    scores_log.append(accumulator)
    
    calc_wm = c_val - k_val
    raw_wm.append(calc_wm)
    
    if accumulator == 5:
        final_signals.append("🟢 STRONG BUY" if c_val > p_high else "❌ NO ENTRY")
        trap_log.append("TREND VALID" if c_val > p_high else "⚠️ BULL TRAP")
    elif accumulator == -5:
        final_signals.append("🔴 STRONG SELL" if c_val < p_low else "🟢 HOLD LONG")
        trap_log.append("TREND VALID" if c_val < p_low else "⚠️ BEAR TRAP")
    else:
        final_signals.append(f"🔄 HOLD (Score: {accumulator})")
        trap_log.append("TREND VALID")

df_predict['d_ML_Signal'] = final_signals
df_predict['Trap_Status'] = trap_log
df_predict['Accumulator_Score'] = scores_log
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_wm)
df_predict['Step_Momentum'] = np.round(apply_non_linear_kalman_momentum(df_predict['Weighted_Momentum'].values))

# Array Layer Arrays Definition 
df_predict['ML_WM_Linear_Prob'] = 0.50
df_predict['ML_WM_NonLinear_Prob'] = 0.50

# Display Matrix
cols_to_show = ['a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Weighted_Momentum', 'Step_Momentum', 'ML_WM_Linear_Prob', 'ML_WM_NonLinear_Prob', 'd_ML_Signal', 'Trap_Status']
st.dataframe(df_predict[cols_to_show].sort_index(ascending=False), use_container_width=True)
