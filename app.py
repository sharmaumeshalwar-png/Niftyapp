import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Bitcoin Ultra-Responsive Engine (1M)", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Minute Ultra-Responsive Engine")
st.write("🎯 **Aapki Perfect Setting:** 1-Minute Candle Tracking + Fixed June 15 Target with Auto-API Fallback")

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

with st.spinner("Aligning 1-Minute Crypto Microstructure Matrices..."):
    # 🔴 yFinance 1m data maximum last 7 days ka deta hai. period="7d" is safe zone.
    df = yf.download("BTC-USD", period="7d", interval="1m")
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    if len(df) == 0:
        st.error("🚨 yFinance Limit Error: Last 7 days 1-minute data limits reached. Please wait 1 minute and re-run.")
        st.stop()

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
    
    # Target Configuration (3-candles forward)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity'], inplace=True)

features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']

# 🛡️ AUTOMATIC FALLBACK API LAYER 
# Agar data availability 15 June ke baad ki hai toh target strictly shift ho jaye auto-pilot par.
requested_date = pd.to_datetime('2026-06-15')
earliest_available_date = df.index.min()

if earliest_available_date > requested_date:
    # Split training on 30% data point of the fetched bucket
    split_idx = int(len(df) * 0.3)
    split_date = df.index[split_idx]
    st.warning(f"⚠️ **yFinance API Limit Warning:** 1-Minute data for June 15 is archived by API. Streaming live metrics using active buffer from {earliest_available_date.strftime('%Y-%m-%d')} onwards.")
else:
    split_date = requested_date

train_mask = df.index < split_date
predict_mask = df.index >= split_date

df_train = df[train_mask].dropna(subset=['Target'])
X_train = df_train[features_matrix]
y_train = df_train['Target']
X_predict = df.loc[predict_mask, features_matrix]

if len(X_predict) == 0:
    st.error("No active matrix predictions streaming. Please check the network block.")
else:
    # 🔴 AAPKI PERFECT LOW SETTING FOR FAST DIFFERENTIATION
    model_flow = RandomForestClassifier(
        n_estimators=150, 
        max_depth=3,            
        min_samples_leaf=1,     
        random_state=42
    )
    model_flow.fit(X_train, y_train)

    probabilities = model_flow.predict_proba(X_predict)
    df_signals = df[predict_mask].copy()
    
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

        # 1. Fresh Signal Rule via Kalman Cross (Aapka strictly verified 60% Filter)
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
        
        # 2. Continuous Monitoring (The Auto-Flip Part!)
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
    
    display_df = display_df.sort_index(ascending=False)
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live 1-Minute Micro-Differentiated Output Window")
    st.dataframe(display_df, use_container_width=True, height=750)
