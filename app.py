import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="BTC Raw K2 Expansion Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 5-Minute Standalone [Raw Kalman 2 Unbounded Expansion Engine]")
st.write("🎯 **Aapki Custom Setting:** Strictly Only BTC 5-Min Data + 50:50 Split + Velocity Useful_VWAP + ML Score $[-5,5]$ + **NEW: Kalman 3 Completely Bypassed, Open Expansion Rule Applied Directly on Weighted_Momentum (Kalman 2)**")

# =====================================================================
# MATHEMATICAL ENGINE (Flexible Kalman Filter Function)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = initial_p  
    q = q_val      
    r = r_val        
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("Bypassing Kalman 3 and Hooking Open Expansion directly to Kalman 2 Waves..."):
    # Bitcoin 5-MINUTE Interval Data (Max period allowed by Yahoo Finance for 5m is 60 days)
    raw_df = yf.download("BTC-USD", period="60d", interval="5m")
    
    if len(raw_df) == 0:
        st.error("YFinance API Timeout or Market Closed. Please refresh the dashboard.")
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

    # Base Matrix Definition (Price Kalman 1 Active)
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0, q_val=0.001, r_val=0.1)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']  # Pure (Close - Kalman)
    
    # -----------------------------------------------------------------
    # INTRADAY BASE VWAP COMPUTATION BLOCK (Resets Every Day)
    # -----------------------------------------------------------------
    typical_price = (df['High'] + df['Low'] + df['a_Close']) / 3
    df['TP_Vol'] = typical_price * df['Volume']
    
    df['Date_Group'] = df.index.date
    cum_tp_vol = df.groupby('Date_Group')['TP_Vol'].cumsum()
    cum_vol = df.groupby('Date_Group')['Volume'].cumsum()
    
    df['VWAP'] = cum_tp_vol / (cum_vol + 1e-10)
    
    # Microstructure Features Space
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Strictly Fixed 25-Candle Directional Lookahead Target Logic
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix + ['Target', 'VWAP'], inplace=True)

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
    st.error("Prediction matrix error. Dataframe split bounds mismatch.")
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
    
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # =====================================================================
    # LIVE TREND-LOCK CIRCUIT (TRIPLE KALMAN SIGNAL PROCESSING)
    # =====================================================================
    final_signals = []
    scores_log = []
    raw_weighted_momentum_log = [] 
    current_state = "HOLD"
    
    accumulator = 0
    MAX_BUCKET = 5     
    MIN_BUCKET = -5    

    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()

    for i in range(len(prob_ups)):
        p_up = prob_ups[i]
        p_down = prob_downs[i]
        c_val = closes[i]
        k_price_val = kalmans_price[i]

        # Raw Accumulator Calculation (ML Bounded standard)
        if p_up >= 0.55:
            accumulator += 1  
        elif p_down >= 0.55:
            accumulator -= 1  
        
        accumulator = max(MIN_BUCKET, min(MAX_BUCKET, accumulator))
        scores_log.append(accumulator)

        # Raw Weighted Momentum (Close - Kalman)
        calc_raw_weighted = c_val - k_price_val
        raw_weighted_momentum_log.append(calc_raw_weighted)

        if accumulator == MAX_BUCKET:
            current_state = "BUY"
            final_signals.append("🟢 STRONG BUY (Max Locked [5/5])")
            
        elif accumulator == MIN_BUCKET:
            current_state = "SELL"
            final_signals.append("🔴 STRONG SELL (Max Locked [-5/-5])")
            
        else:
            if current_state == "BUY":
                if accumulator > 0:
                    final_signals.append(f"🟢 HOLD BUY | Points Decreasing (Score: {accumulator})")
                else:
                    final_signals.append(f"⚠️ BUY CRITICAL | Reversal Warning (Score: {accumulator})")
                    
            elif current_state == "SELL":
                if accumulator < 0:
                    final_signals.append(f"🔴 HOLD SELL | Points Increasing (Score: {accumulator})")
                else:
                    final_signals.append(f"⚠️ SELL CRITICAL | Reversal Warning (Score: {accumulator})")
                    
            else:
                final_signals.append(f"⚪ NEUTRAL | Building Conviction (Score: {accumulator})")

    # Mapping secure array data back to pandas
    df_predict['d_ML_Signal'] = final_signals
    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 

    # [Kalman 2 Execution] Runs on Raw_Weighted_Momentum (P=0.50 Standard Tracking)
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(
        df_predict['Raw_Weighted_Momentum'].values, 
        initial_p=0.50, q_val=0.001, r_val=0.1
    )

    # -----------------------------------------------------------------
    # VELOCITY DELTA INTERACTION ON VWAP MATRICES
    # -----------------------------------------------------------------
    df_predict['Useful_VWAP'] = df_predict['VWAP'] * (df_predict['Prob_Up'].shift(1) - df_predict['Prob_Up'])
    df_predict['Useful_VWAP'] = df_predict['Useful_VWAP'].fillna(0)
    
    # NOTE: Kalman 3 layers have been stripped out. 
    # We apply the stretch rules directly on df_predict['Weighted_Momentum'] (Kalman 2)

    # -----------------------------------------------------------------
    # 🎯 NEW CHASSIS CORE: UNBOUNDED EXPANSION ENGINE DIRECT ON KALMAN 2
    # -----------------------------------------------------------------
    k2_values = df_predict['Weighted_Momentum'].to_numpy()
    k2_unbounded_log = []
    unbounded_accumulator = 0  # Anchor index baseline

    for idx in range(len(k2_values)):
        if idx == 0:
            k2_unbounded_log.append(0)
            continue
            
        # Directional checks directly tracked on Kalman 2 (Weighted Momentum Vector)
        if k2_values[idx] > k2_values[idx - 1]:
            unbounded_accumulator += 1   # Infinite expansion to the upside
        elif k2_values[idx] < k2_values[idx - 1]:
            unbounded_accumulator -= 1   # Infinite expansion to the downside
            
        # Hard limits completely bypassed to track pure structural stretch & reversals
        k2_unbounded_log.append(unbounded_accumulator)
        
    df_predict['K2_Open_Score'] = k2_unbounded_log

    # Display Configuration
    clean_display_cols = ['a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Weighted_Momentum', 'K2_Open_Score', 'd_ML_Signal']
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    display_df['Accumulator_Score'] = display_df['Accumulator_Score'].astype(int)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    display_df['K2_Open_Score'] = display_df['K2_Open_Score'].astype(int)
    
    # Inverting framework to see latest 5-min intervals on top rows
    display_df = display_df.sort_index(ascending=False)
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live 5-Minute BTC Standalone Engine (Raw K2 Open Expansion Wave Matrix)")
    st.dataframe(display_df, use_container_width=True, height=750)
