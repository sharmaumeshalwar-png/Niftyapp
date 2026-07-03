import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty 1H 50:50 Engine", layout="wide")
st.title("⚡ Nifty 50 Live 1-Hour Engine (Equal Split Layout)")
st.write("🎯 **Aapki Perfect Setting:** 2-Year Hourly Horizon + Strict 50:50 Matrix Balance")

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

with st.spinner("Aligning Responsive Nifty 1-Hour Microstructure Matrices..."):
    # NIFTY 1-HOUR DATA (2 Years Window)
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    
    if len(raw_df) == 0:
        st.error("YFinance API Timeout or NSE Market Closed. Please refresh the dashboard.")
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
    
    # Target Configuration (3 Hours Lookahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix, inplace=True)

# =====================================================================
# DYNAMIC SPLIT ENGINE (🌟 AAPKI REQUIREMENT: Strict 50:50 Ratio)
# =====================================================================
# Total rows ka exact 50% split points index
split_idx = int(len(df) * 0.50)

# Pehla 50% historical data direct Training Matrix me
df_train = df.iloc[:split_idx].dropna(subset=['Target'])
X_train = df_train[features_matrix]
y_train = df_train['Target']

# Aakhri 50% data (Current Ticks tak) Prediction Matrix me
df_predict = df.iloc[split_idx:]
X_predict = df_predict[features_matrix]

if len(X_predict) == 0:
    st.error("Prediction matrix error. Waiting for market data...")
else:
    # Aggressive Low-Parameter Model Settings
    model_flow = RandomForestClassifier(
        n_estimators=150, 
        max_depth=3,            
        min_samples_leaf=1,     
        random_state=42
    )
    model_flow.fit(X_train, y_train)

    # Probabilities derivation
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
    clean_display_cols =
