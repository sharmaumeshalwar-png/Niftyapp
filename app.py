import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Probability Divergence", layout="wide")
st.title("📊 Nifty 50 Live 1-Hour Probability Divergence Engine")
st.write("🎯 **Advanced Spread Logic:** Comparing 1-Candle Micro Prob vs 5-Candle Macro Prob directly on Weighted Momentum.")

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

with st.spinner("🔮 Initializing Dual-Probability Neural Engines..."):
    # Safe single-layer column download
    raw_df = yf.download("^NSEI", period="2y", interval="1h", multi_level_index=False)
    
    if raw_df.empty:
        raw_df = yf.download("^NSEI", period="1mo", interval="1h", multi_level_index=False)
        
    if raw_df.empty:
        st.error("YFinance API Timeout or Indian Market Closed. Please refresh the dashboard.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].ffill()

    df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)
    df.index = pd.to_datetime(df.index)

    # Base Matrix Definition
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=100.0)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']  
    
    # Generate Weighted Momentum Wave
    df['Weighted_Momentum'] = apply_kalman_filter_custom(df['c_Combined'].values, initial_p=0.50)
    
    # Structural Features Matrix
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    
    # 🎯 DUAL TARGET DEFINITION
    df['Target_1'] = np.where(df['Weighted_Momentum'] > df['Weighted_Momentum'].shift(1), 1, 0)
    df['Target_5'] = np.where(df['Weighted_Momentum'] > df['Weighted_Momentum'].shift(5), 1, 0)
    
    df_clean = df.replace([np.inf, -np.inf], np.nan).copy()

# Dynamic Split Engine (50:50 Split)
split_idx = int(len(df_clean) * 0.50)

df_train = df_clean.iloc[:split_idx].copy()
df_train.dropna(subset=features_matrix + ['Target_1', 'Target_5'], inplace=True)

X_train = df_train[features_matrix]
y_train_1 = df_train['Target_1']
y_train_5 = df_train['Target_5']

df_predict = df_clean.iloc[split_idx:].copy()
df_predict.dropna(subset=features_matrix, inplace=True) 

X_predict = df_predict[features_matrix]

if len(X_predict) == 0 or len(X_train) == 0:
    st.error(f"⚠️ Data size insufficient for split.")
else:
    # 🌲 Model 1: Train for 1-Candle Shift Momentum
    model_1 = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42)
    model_1.fit(X_train, y_train_1)
    
    # 🌲 Model 5: Train for 5-Candle Shift Momentum
    model_5 = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42)
    model_5.fit(X_train, y_train_5)

    # Generate Parallel Probabilities
    prob_1 = model_1.predict_proba(X_predict)
    prob_5 = model_5.predict_proba(X_predict)
    
    df_predict['Prob1_Up'] = prob_1[:, 1]
    df_predict['Prob5_Up'] = prob_5[:, 1]
    
    # 🆕 MATHEMATICAL SPREAD: Prob 1 Up - Prob 5 Up
    df_predict['Prob_Spread'] = df_predict['Prob1_Up'] - df_predict['Prob5_Up']

    # Price Action Columns
    df_predict['Prev_High'] = df_predict['High'].shift(1)
    df_predict['Prev_Low'] = df_predict['Low'].shift(1)

    # Live Circuit Log Generation
    final_signals = []
    scores_log = []
    current_state = "HOLD"
    accumulator = 0

    spreads = df_predict['Prob_Spread'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    prev_highs = df_predict['Prev_High'].to_numpy()
    prev_lows = df_predict['Prev_Low'].to_numpy()
    weighted_moms = df_predict['Weighted_Momentum'].to_numpy()

    for i in range(len(spreads)):
        spr = spreads[i]
        c_val = closes[i]
        p_high = prev_highs[i] if not np.isnan(prev_highs[i]) else c_val
        p_low = prev_lows[i] if not np.isnan(prev_lows[i]) else c_val
        w_mom = weighted_moms[i]

        # Accumulator strictly reacts to the Directional Spread Filter
        if spr > 0.02: accumulator += 1       # Short-term faster than Medium-term
        elif spr < -0.02: accumulator -= 1    # Short-term losing velocity
        else:
            # Convergence mapping towards neutral if spreads squeeze
            if accumulator > 0: accumulator -= 1
            elif accumulator < 0: accumulator += 1
            
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)

        if accumulator == 5:
            current_state = "BUY"
            if c_val > p_high and w_mom > 0: 
                final_signals.append("🟢 STRONG SPREAD BUY (Velocity Confirmed)")
            else: 
                final_signals.append("❌ NO ENTRY (Trap Filter Active)")
        elif accumulator == -5:
            current_state = "SELL"
            if c_val < p_low and w_mom < 0: 
                final_signals.append("🔴 STRONG SPREAD SELL (Velocity Confirmed)")
            else: 
                final_signals.append("🟢 HOLD LONG (No Short Entry)")
        else:
            if current_state == "BUY":
                if accumulator > 0: final_signals.append(f"🟢 HOLD BUY | Spread Weakening ({accumulator})")
                else: final_signals.append(f"⚠️ MOMENTUM FLIP | Exit Buy ({accumulator})")
            elif current_state == "SELL":
                if accumulator < 0: final_signals.append(f"🔴 HOLD SELL | Spread Recovering ({accumulator})")
                else: final_signals.append(f"⚠️ MOMENTUM FLIP | Exit Sell ({accumulator})")
            else:
                final_signals.append(f"⚪ NEUTRAL | Squeezing Range ({accumulator})")

    df_predict['d_ML_Signal'] = final_signals
    df_predict['Accumulator_Score'] = scores_log

    # Table View Layout Configuration
    clean_display_cols = [
        'a_Close', 'Weighted_Momentum', 'Prob1_Up', 'Prob5_Up', 
        'Prob_Spread', 'Accumulator_Score', 'd_ML_Signal'
    ]
    
    display_df = df_predict[clean_display_cols].copy().sort_index(ascending=False)
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(4)
    display_df['Prob1_Up'] = display_df['Prob1_Up'].round(3)
    display_df['Prob5_Up'] = display_df['Prob5_Up'].round(3)
    display_df['Prob_Spread'] = display_df['Prob_Spread'].round(4)
    display_df['Accumulator_Score'] = display_df['Accumulator_Score'].astype(int)
    
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live 1-Hour Nifty Probability Divergence Dashboard")
    st.dataframe(display_df, use_container_width=True, height=750)
