import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty 50 Ultra-Responsive Engine", layout="wide")
st.title("⚡ Nifty 50 Live Dynamic-Flip & Low-Parameter Engine")
st.write("🎯 **Aapki Perfect Setting:** Pure 50-Day Data Horizon + Strict 80:20 Split Applied on Nifty Index")

# =====================================================================
# MATHEMATICAL ENGINE (Kalman Filter 0.001)
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

with st.spinner("Aligning Responsive Nifty Market Microstructure Matrices..."):
    # 🌟 NIFTY 50 INDEX TICKER OVERRIDE (^NSEI)
    raw_df = yf.download("^NSEI", period="50d", interval="5m")
    
    if len(raw_df) == 0:
        st.error("YFinance API Timeout or NSE Market Closed. Please refresh the dashboard.")
        st.stop()
        
    # Structure ko strict 1D columns me convert karna (Line 104 Fix)
    df = pd.DataFrame(index=raw_df.index)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            if isinstance(raw_df[col], pd.DataFrame):
                df[col] = raw_df[col].iloc[:, 0]  # Take first column if multi-index
            else:
                df[col] = raw_df[col]

    df.index = pd.to_datetime(df.index)

    # Base Matrix Definition
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
    
    # Target Configuration (Future 3 candles lookahead = 15 Mins forecast)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix, inplace=True)

# =====================================================================
# DYNAMIC SPLIT ENGINE (Strict 80:20 Ratio)
# =====================================================================
split_idx = int(len(df) * 0.80)

# 80% Training Matrix (Historical Nifty Ticks)
df_train = df.iloc[:split_idx].dropna(subset=['Target'])
X_train = df_train[features_matrix]
y_train = df_train['Target']

# 20% Live Prediction Matrix (Latest Nifty Ticks)
df_predict = df.iloc[split_idx:]
X_predict = df_predict[features_matrix]

if len(X_predict) == 0:
    st.error("Prediction matrix calculation error. Waiting for Indian market trading hours...")
else:
    # Aggressive Low-Parameter Model Settings
    model_flow = RandomForestClassifier(
        n_estimators=150, 
        max_depth=3,            
        min_samples_leaf=1,     
        random_state=42
    )
    model_flow.fit(X_train, y_train)

    # Probabilities processing
    probabilities = model_flow.predict_proba(X_predict)
    df_signals = df_predict.copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # =====================================================================
    # LIVE DYNAMIC AUTO-FLIP CIRCUIT 🛡️
    # =====================================================================
    final_signals = []
    current_state = "HOLD"

    sign_changes = df_signals['Sign_Change'].values
    prob_ups = df_signals['Prob_Up'].values
    prob_downs = df_signals['Prob_Down'].values

    for i in range(len(df_signals)):
        sc = sign_changes[i]
        p_up = prob_ups[i]
        p_down = prob_downs[i]

        if sc == 1:
            if p_up >= 0.60:  
                current_state = "BUY"
                final_signals.append("🟢 INSTITUTIONAL BUY (Confirmed)")
            elif p_down >= 0.60:
                current_state = "SELL"
                final_signals.append("🔴 INSTITUTIONAL SELL (Confirmed)")
            else:
                current_state = "HOLD"
                final_signals.append("⚪ HOLD")
        else:
            if current_state == "BUY":
                if p_down > 0.52 or p_up < 0.50:
                    current_state = "SELL"
                    final_signals.append("🔴 SYSTEM AUTO-FLIP (SELL / Exit Buy)")
                else:
                    final_signals.append("🟢 HOLD BUY TREND")
            elif current_state == "SELL":
                if p_up > 0.52 or p_down < 0.50:
                    current_state = "BUY"
                    final_signals.append("🟢 SYSTEM AUTO-FLIP (BUY / Exit Sell)")
                else:
                    final_signals.append("🔴 HOLD SELL TREND")
            else:
                final_signals.append("⚪ HOLD")

    df_signals['d_ML_Signal'] = final_signals

    # Display Configuration
    clean_display_cols = ['a_Close', 'b_Kalman', 'Prob_Up', 'Prob_Down', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    
    # Newest ticks displayed on top
    display_df = display_df.sort_index(ascending=False)
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live Micro-Differentiated Nifty Engine (50D Horizon - 80:20 Split)")
    st.dataframe(display_df, use_container_width=True, height=750)
