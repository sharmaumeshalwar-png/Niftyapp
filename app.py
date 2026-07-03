import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Bitcoin 1H Stable Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Hour Engine (Low-Flip Configuration)")
st.write("🎯 **Aapki Perfect Setting:** 2-Year Hourly Horizon + Strict 50:50 Split + Stable Auto-Flip Circuit")

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
    
    # Past 10-Candle Target (Past-Locked Model)
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(10), 1, 0)
    
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
    # Aggressive Low-Parameter Model Settings
    model_flow = RandomForestClassifier(
        n_estimators=150, 
        max_depth=3,            
        min_samples_leaf=1,     
        random_state=42
    )
    model_flow.fit(X_train, y_train)

    # Bulletproof Slicing Allocation
    probabilities = model_flow.predict_proba(X_predict)
    
    df_predict.loc[:, 'Prob_Down'] = probabilities[:, 0]
    df_predict.loc[:, 'Prob_Up'] = probabilities[:, 1]

    # =====================================================================
    # LIVE DYNAMIC AUTO-FLIP CIRCUIT (🌟 REDUCED FLIP FILTERS)
    # =====================================================================
    final_signals = []
    current_state = "HOLD"

    sign_changes = df_predict['Sign_Change'].values
    prob_ups = df_predict['Prob_Up'].values
    prob_downs = df_predict['Prob_Down'].values

    for i in range(len(df_predict)):
        sc = sign_changes[i]
        p_up = prob_ups[i]
        p_down = prob_downs[i]

        if sc == 1:
            # 🌟 Filter Tighten: Fresh crossing requires 65% conviction now
            if p_up >= 0.65:  
                current_state = "BUY"
                final_signals.append("🟢 INSTITUTIONAL BUY (Confirmed)")
            elif p_down >= 0.65:
                current_state = "SELL"
                final_signals.append("🔴 INSTITUTIONAL SELL (Confirmed)")
            else:
                # Agar state strong nahi hai toh purani state hold rakho, bina flip kiye
                if current_state == "BUY":
                    final_signals.append("🟢 HOLD BUY TREND")
                elif current_state == "SELL":
                    final_signals.append("🔴 HOLD SELL TREND")
                else:
                    final_signals.append("⚪ HOLD")
        else:
            if current_state == "BUY":
                # 🌟 Noise Filter: Auto-flip tabhi hoga jab buyers power < 45% drop ho ya sellers > 55% ho
                if p_down > 0.55 or p_up < 0.45:
                    current_state = "SELL"
                    final_signals.append("🔴 SYSTEM AUTO-FLIP (SELL / Exit Buy)")
                else:
                    final_signals.append("🟢 HOLD BUY TREND")
            elif current_state == "SELL":
                # 🌟 Noise Filter: Auto-flip tabhi hoga jab sellers power < 45% drop ho ya buyers > 55% ho
                if p_up > 0.55 or p_down < 0.45:
                    current_state = "BUY"
                    final_signals.append("🟢 SYSTEM AUTO-FLIP (BUY / Exit Sell)")
                else:
                    final_signals.append("🔴 HOLD SELL TREND")
            else:
                final_signals.append("⚪ HOLD")

    df_predict.loc[:, 'd_ML_Signal'] = final_signals

    # Display Configuration
    clean_display_cols = ['a_Close', 'b_Kalman', 'Prob_Up', 'Prob_Down', 'd_ML_Signal']
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    
    # Newest hourly candles sorted on top
    display_df = display_df.sort_index(ascending=False)
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live 1-Hour Bitcoin Engine (Strict Low-Flip Engine)")
    st.dataframe(display_df, use_container_width=True, height=750)
