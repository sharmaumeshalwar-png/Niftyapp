import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="BTC Standalone 5M Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 5-Minute Standalone Triple Kalman [Scalping Alpha Engine]")
st.write("🎯 **Aapki Custom Setting:** Strictly Only BTC 5-Min Data + Strict 50:50 Train-Test Split + Price Kalman + Fixed 25-Candle Target Window + Pure Raw Accumulator + Parallel Dual Momentum (K2: Smooth Momentum | K3: Kalman 0.50 on Alpha Discovery Vector)")

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

with st.spinner("Aligning 5-Minute Triple Kalman Bitcoin Microstructure Matrices..."):
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
    # INTRADAY TRUE VWAP COMPUTATION BLOCK (Resets Every Day)
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

        # Raw Accumulator Calculation
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

    # [Kalman 2] Runs on Raw_Weighted_Momentum (P=0.50 standard setting)
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(
        df_predict['Raw_Weighted_Momentum'].values, 
        initial_p=0.50, q_val=0.001, r_val=0.1
    )

    # -----------------------------------------------------------------
    # [Kalman 3] ADVANCED PROBABILITY BIAS DISCOVERY CORE (5M INTERVAL)
    # -----------------------------------------------------------------
    # 1. Structural Deviation Vector (Kalman 2 - VWAP)
    df_predict['K2_Minus_VWAP'] = df_predict['Weighted_Momentum'] - df_predict['VWAP']
    
    # 2. Absolute Probability Bias: |Prob_Up - Prob_Down|
    df_predict['Abs_Prob_Bias'] = (df_predict['Prob_Up'] - df_predict['Prob_Down']).abs()
    
    # 3. Dynamic Alpha Discovery Vector Generation
    df_predict['Alpha_Discovery_Base'] = df_predict['K2_Minus_VWAP'] * df_predict['Abs_Prob_Bias']
    
    # 4. Applying standard Kalman Filter with initial_p=0.50 on the Discovery vector
    df_predict['Triple_Kalman_Discovery'] = apply_kalman_filter_custom(
        df_predict['Alpha_Discovery_Base'].values, 
        initial_p=0.50, q_val=0.001, r_val=0.1
    )

    # Display Configuration
    clean_display_cols = ['a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Weighted_Momentum', 'Triple_Kalman_Discovery', 'd_ML_Signal']
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    display_df['Accumulator_Score'] = display_df['Accumulator_Score'].astype(int)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    display_df['Triple_Kalman_Discovery'] = display_df['Triple_Kalman_Discovery'].round(2) 
    
    # Sorting to show most recent 5-min intervals on top rows
    display_df = display_df.sort_index(ascending=False)
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live 5-Minute BTC Standalone Engine (50:50 Parallel Discovery Matrix)")
    st.dataframe(display_df, use_container_width=True, height=750)
