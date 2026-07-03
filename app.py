import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Bitcoin 1H Trend Tracker", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Hour Trend-Lock Engine")
st.write("🎯 **Aapki Perfect Setting:** 2-Year Hourly Horizon + Strict 50:50 Split + Kalman Price + Accumulator Bucket (Zero Flip)")

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
    
    # Past 10-Candle Target
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
    # RandomForest Model Training
    model_flow = RandomForestClassifier(
        n_estimators=150, 
        max_depth=3,            
        min_samples_leaf=1,     
        random_state=42
    )
    model_flow.fit(X_train, y_train)

    # Raw Probabilities Prediction (No Filters Applied)
    probabilities = model_flow.predict_proba(X_predict)
    df_predict.loc[:, 'Prob_Down'] = probabilities[:, 0]
    df_predict.loc[:, 'Prob_Up'] = probabilities[:, 1]

    # =====================================================================
    # 🌟 LIVE TREND-LOCK CIRCUIT (Accumulator Bucket Logic)
    # =====================================================================
    final_signals = []
    current_state = "HOLD"
    
    # Bucket settings (Threshold limits)
    accumulator = 0
    MAX_BUCKET = 5     # Kitne points par trend lock hoga
    MIN_BUCKET = -5    # Kitne negative points par reverse lock hoga

    prob_ups = df_predict['Prob_Up'].values
    prob_downs = df_predict['Prob_Down'].values
    sign_changes = df_predict['Sign_Change'].values

    for i in range(len(df_predict)):
        p_up = prob_ups[i]
        p_down = prob_downs[i]
        sc = sign_changes[i]

        # Points generation based on raw conviction
        if p_up >= 0.55:
            accumulator += 1  # Bullish momentum building
        elif p_down >= 0.55:
            accumulator -= 1  # Bearish momentum building
        
        # Keep accumulator within boundaries
        accumulator = max(MIN_BUCKET, min(MAX_BUCKET, accumulator))

        # Trend Decision Logic
        if accumulator >= MAX_BUCKET:
            if current_state != "BUY":
                current_state = "BUY"
                final_signals.append("🟢 INSTITUTIONAL BUY (Trend Locked)")
            else:
                final_signals.append("🟢 HOLD BUY TREND")
                
        elif accumulator <= MIN_BUCKET:
            if current_state != "SELL":
                current_state = "SELL"
                final_signals.append("🔴 INSTITUTIONAL SELL (Trend Locked)")
            else:
                final_signals.append("🔴 HOLD SELL TREND")
                
        else:
            # Jab accumulator beech mein ho (Chop zone), toh purani state maintain rakho
            if current_state == "BUY":
                final_signals.append("🟢 HOLD BUY TREND")
            elif current_state == "SELL":
                final_signals.append("🔴 HOLD SELL TREND")
            else:
                final_signals.append("⚪ HOLD (Building Conviction)")

    df_predict.loc[:, 'd_ML_Signal'] = final_signals
    df_predict.loc[:, 'Accumulator_Score'] = accumulator  # Monitor ke liye column add kiya

    # Display Configuration
    clean_display_cols = ['a_Close', 'b_Kalman', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'd_ML_Signal']
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    
    # Sorting to get latest ticks on top
    display_df = display_df.sort_index(ascending=False)
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live 1-Hour Bitcoin Engine (Zero-Flip Accumulator Framework)")
    st.dataframe(display_df, use_container_width=True, height=750)
