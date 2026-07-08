import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty 1-Candle Fast Engine", layout="wide")
st.title("⚡ Nifty 50 Live 1-Hour Ultra-Reactive 1-Candle Engine")
st.write("🎯 **Micro-Shift Framework:** Only Nifty 50 Data + Trusted RandomForest Core + Strictly 1-Candle Target Window + Smooth Distance Price Kalman")

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

with st.spinner("🚀 Aligning 1-Candle Micro-Target Neural Matrix..."):
    # Fail-safe data download
    raw_df = yf.download("^NSEI", period="2y", interval="1h", group_by='column')
    
    if raw_df.empty:
        raw_df = yf.download("^NSEI", period="1mo", interval="1h")
        
    if raw_df.empty:
        st.error("YFinance API Timeout or Indian Market Closed. Please refresh the dashboard.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            if isinstance(raw_df[col], pd.DataFrame):
                df[col] = raw_df[col].iloc[:, 0].ffill()
            else:
                df[col] = raw_df[col].ffill()

    df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)
    df.index = pd.to_datetime(df.index)

    # Base Matrix Definition
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

    # 🎯 TARGET CHANGE: 25 candle hata kar ab sirf agali 1 candle check hogi
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(1), 1, 0)
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
    # 🌲 Puraana trusted RandomForest structure jaisa aapne bola
    model_flow = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42)
    model_flow.fit(X_train, y_train)

    # Fast Probabilities Generation
    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # Price Action Columns
    df_predict['Prev_High'] = df_predict['High'].shift(1)
    df_predict['Prev_Low'] = df_predict['Low'].shift(1)

    # Live Circuit Log Generation
    final_signals = []
    scores_log = []
    raw_weighted_momentum_log = [] 
    trap_status_log = [] 
    current_state = "HOLD"
    accumulator = 0

    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()
    prev_highs = df_predict['Prev_High'].to_numpy()
    prev_lows = df_predict['Prev_Low'].to_numpy()

    for i in range(len(prob_ups)):
        p_up = prob_ups[i]
        p_down = prob_downs[i]
        c_val = closes[i]
        k_price_val = kalmans_price[i]
        p_high = prev_highs[i] if not np.isnan(prev_highs[i]) else c_val
        p_low = prev_lows[i] if not np.isnan(prev_lows[i]) else c_val

        # Accumulator Logic (Will respond sharply to 1-candle shifts)
        if p_up >= 0.55: accumulator += 1  
        elif p_down >= 0.55: accumulator -= 1  
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)

        calc_raw_weighted = c_val - k_price_val
        raw_weighted_momentum_log.append
