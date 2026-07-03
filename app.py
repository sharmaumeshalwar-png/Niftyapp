import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Solana Macro 1H Engine", layout="wide")
st.title("⚡ Solana (SOL) 1-Hour Macro-Learning Engine")
st.write("🎯 **Aapki High-Data Setting:** 1-Hour Candles + 1.5 Years Deep Learning + Fixed 1 Jan 2026 Prediction")

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

with st.spinner("Downloading 2 Years Macro Data & Training Model (July 2024 Onwards)..."):
    # 🔴 1-Hour interval ke sath period="2y" kiya hai taaki 1 saal se upar ka deep data mile
    df = yf.download("SOL-USD", period="2y", interval="1h")
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    if len(df) == 0:
        st.error("YFinance API Timeout. Max data window mismatch. Please refresh.")
        st.stop()

    df.index = pd.to_datetime(df.index)
    # Ensure index is timezone naive for clean string comparison
    df.index = df.index.tz_localize(None)

    # Base Matrix Definition
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # Microstructure Features (Hourly Standard)
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Target Configuration (3-candles forward in 1-Hour space = 3 Hours lookahead trend)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity'], inplace=True)

features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']

# 📅 AAPKI REQUESTED TIMELINE: 1 Jan 2026 se pehle ka sab seekhne me jayega, baad ka prediction me.
train_mask = df.index < '2026-01-01'
predict_mask = df.index >= '2026-01-01'

df_train = df[train_mask].dropna(subset=['Target'])
X_train = df_train[features_matrix]
y_train = df_train['Target']
X_predict = df.loc[predict_mask, features_matrix]

if len(X_predict) == 0:
    st.error("No active matrix predictions streaming from January 1, 2026 onwards. Please re-verify data chunk.")
else:
    # 🔴 AAPKI PERFECT ORIGINAL LOW SETTING (Strictly Same Nodes)
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
    # LIVE DYNAMIC AUTO-FLIP CIRCUIT (Original Same Copy Setup) 🛡️
    # =====================================================================
    final_signals = []
    current_state = "HOLD"

    sign_changes = df_signals['Sign_Change'].values
    prob_ups = df_signals['Prob_Up'].values
    prob_downs = df_signals['Prob_Down'].values
    total_rows = len(df_signals)

    for i in range(total_rows):
        sc = sign_changes[i]
        p_up = prob_ups[i]
        p_down = prob_downs[i]

        # 1. Fresh Signal Rule via Kalman Cross (Original Strict 60% Barrier)
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

    # Display Configuration (Rounded to 2 decimals for absolute clarity)
    clean_display_cols = ['a_Close', 'b_Kalman', 'Prob_Up', 'Prob_Down', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    
    display_df = display_df.sort_index(ascending=False)
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live 1-Hour Solana Dashboard (Predicting from 1st January 2026 Onwards)")
    st.dataframe(display_df, use_container_width=True, height=750)
