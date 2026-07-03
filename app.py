import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Bitcoin 1H Descriptive Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Hour Micro-Trend Engine")
st.write("🎯 **Aapki Perfect Setting:** 2-Year Hourly Horizon + Strict 50:50 Split + Full Points Slicing Display")

# =====================================================================
# MATHEMATICAL ENGINE (Kalman Filter ONLY for Price)
# =====================================================================
def apply_kalman_filter_strict(price_array):
    if len(price_array) == 0:
        return []
    x = price_array[0]
    p = 50.0  
    q = 0.001  
    r = 0.1    
    filtered_prices = []
    for z in price_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_prices.append(x)
    return filtered_prices

with st.spinner("Aligning Responsive Bitcoin 1-Hour Microstructure Matrices..."):
    # BTC-USD Hourly 2 Years Window
    raw_df = yf.download("BTC-USD", period="2y", interval="1h")
    
    if len(raw_df) == 0:
        st.error("YFinance API Timeout. Please refresh the dashboard.")
        st.stop()
        
    # MultiIndex Framework Elimination
    df = pd.DataFrame(index=raw_df.index)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            if isinstance(raw_df[col], pd.DataFrame):
                df[col] = raw_df[col].iloc[:, 0]
            else:
                df[col] = raw_df[col]

    df.index = pd.to_datetime(df.index)

    # Base Matrix Definition (Price Kalman Active)
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # Microstructure Features
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Past 25-Candle Target
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix + ['Target'], inplace=True)

# =====================================================================
# DYNAMIC SPLIT ENGINE (Strict 50:50 Ratio)
# =====================================================================
split_idx = int(len(df) * 0.50)

df_train = df.iloc[:split_idx]
X_train = df_train[features_matrix].copy()
y_train = df_train['Target'].copy()

df_predict = df.iloc[split_idx:].copy()
X_predict = df_predict[features_matrix].copy()

if len(X_predict) == 0:
    st.error("Prediction matrix error. Waiting for market data ticks...")
else:
    # RandomForest Model Training
    model_flow = RandomForestClassifier(
        n_estimators=150, 
        max_depth=3,            
        min_samples_leaf=1,     
        random_state=42
    )
    model_flow.fit(X_train, y_train)

    # Raw Probabilities Prediction
    probabilities = model_flow.predict_proba(X_predict)
    df_predict.loc[:, 'Prob_Down'] = probabilities[:, 0]
    df_predict.loc[:, 'Prob_Up'] = probabilities[:, 1]

    # =====================================================================
    # 🌟 LIVE TREND-LOCK CIRCUIT (Descriptive Step-by-Step Display)
    # =====================================================================
    final_signals = []
    scores_log = []
    current_state = "HOLD"
    
    accumulator = 0
    MAX_BUCKET = 5     
    MIN_BUCKET = -5    

    prob_ups = df_predict['Prob_Up'].values
    prob_downs = df_predict['Prob_Down'].values

    for i in range(len(df_predict)):
        p_up = prob_ups[i]
        p_down = prob_downs[i]

        # Points Addition/Reduction
        if p_up >= 0.55:
            accumulator += 1  
        elif p_down >= 0.55:
            accumulator -= 1  
        
        # Keep within boundaries
        accumulator = max(MIN_BUCKET, min(MAX_BUCKET, accumulator))
        scores_log.append(accumulator)

        # 🌟 AAPKI REQUIREMENT: Pure flow ko level wise display karna (5, 4, 3, 2, 1...)
        if accumulator == MAX_BUCKET:
            current_state = "BUY"
            final_signals.append("🟢 STRONG BUY TREND (Max Locked [5/5])")
            
        elif accumulator == MIN_BUCKET:
            current_state = "SELL"
            final_signals.append("🔴 STRONG SELL TREND (Max Locked [-5/-5])")
            
        else:
            # Jab score decrease ya increase ho raha ho (Between -4 and +4)
            if current_state == "BUY":
                if accumulator > 0:
                    final_signals.append(f"🟢 HOLD BUY | Trend Softening (Score: {accumulator})")
                else:
                    final_signals.append(f"⚠️ BUY TREND CRITICAL | Reversal Warning (Score: {accumulator})")
                    
            elif current_state == "SELL":
                if accumulator < 0:
                    final_signals.append(f"🔴 HOLD SELL | Trend Softening (Score: {accumulator})")
                else:
                    final_signals.append(f"⚠️ SELL TREND CRITICAL | Reversal Warning (Score: {accumulator})")
                    
            else:
                final_signals.append(f"⚪ NEUTRAL | Building Conviction (Score: {accumulator})")

    df_predict.loc[:, 'd_ML_Signal'] = final_signals
    df_predict.loc[:, 'Accumulator_Score'] = scores_log  

    # Display Configuration
    clean_display_cols = ['a_Close', 'b_Kalman', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'd_ML_Signal']
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['Prob_Up'] =
