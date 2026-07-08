import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Hint Mapping Engine", layout="wide")
st.title("📊 Nifty 50 Live 1-Hour Standalone Engine [Exact Hint Sheet Integrated]")
st.write("🎯 **Aapki Custom Setting:** Strictly Only Nifty 50 Index Data + Price Kalman + Fixed 25-Candle Target Window + **Aapki Written Sheet Hints (CE/PE Sell Constraints)** + Kalman Momentum (P=0.50)")

# =====================================================================
# AAPKI SHEET KA EXACT DATA MAPPER (Soft-Matching Enabled)
# =====================================================================
hint_sheet_data = {
    # Left Column (2025)
    "2025-07-04": "CE_SELL", "2025-07-08": "PE_SELL",
    "2025-07-09": "CE_SELL", "2025-02-18": "PE_SELL",
    "2025-08-26": "CE_SELL", "2025-09-04": "PE_SELL",
    "2025-09-08": "PE_SELL", "2025-09-25": "CE_SELL", 
    "2025-10-06": "PE_SELL", "2025-11-04": "CE_SELL", 
    "2025-11-11": "PE_SELL", "2025-11-25": "CE_SELL", 
    "2025-11-26": "PE_SELL", "2025-12-02": "CE_SELL", 
    "2025-12-05": "PE_SELL", "2025-12-08": "CE_SELL", 
    "2025-12-22": "PE_SELL", "2025-12-24": "CE_SELL",
    # Transition to 2026 (Left Column Bottom)
    "2026-01-01": "PE_SELL",
    
    # Right Column (2025-2026 standard mappings)
    "2025-01-01": "CE_SELL", "2025-01-02": "PE_SELL", 
    "2025-01-06": "CE_SELL", "2025-02-03": "PE_SELL", 
    "2025-02-05": "CE_SELL", "2025-02-06": "CE_SELL",
    "2025-07-06": "PE_SELL", "2025-02-13": "CE_SELL",
    "2025-02-16": "PE_SELL", "2025-02-19": "CE_SELL",
    "2025-02-23": "CE_SELL", "2025-02-24": "CE_SELL",
    "2025-04-08": "PE_SELL", "2025-05-12": "CE_SELL",
    "2025-05-14": "PE_SELL", "2025-05-29": "CE_SELL",
    "2025-06-12": "PE_SELL"
}

# =====================================================================
# MATHEMATICAL ENGINE (Kalman Filter Function)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0):
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = initial_p  
    q = 0.001      # Process noise
    r = 0.1        # Measurement noise
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("Aligning Your Hint Sheet Matrices onto Live Nifty 50 Grid..."):
    # Download raw Nifty Data (Using period '2y' to guarantee baseline recovery)
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    
    if len(raw_df) == 0:
        st.error("YFinance API Mismatch or Market Closed. Showing structural backup frame.")
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

    # Base Matrix Pricing Structure
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']  
    
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # Microstructure Calculations
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Target Horizon Mapping Lookahead Fixed at 25-Candles
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
    st.error("Prediction matrix boundary mismatch tracking error.")
else:
    # Model Training Sequence
    model_flow = RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=1, random_state=42)
    model_flow.fit(X_train, y_train)

    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # =====================================================================
    # AAPKI HINT SHEET MAPPING LOGIC AND OVERRIDES
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
    timestamps = df_predict.index

    for i in range(len(prob_ups)):
        p_up = prob_ups[i]
        p_down = prob_downs[i]
        c_val = closes[i]
        k_price_val = kalmans_price[i]
        
        # Exact date check system without restrictive intraday minute traps
        current_date_str = timestamps[i].strftime('%Y-%m-%d')

        hint_match = hint_sheet_data.get(current_date_str, None)

        if hint_match is not None:
            if hint_match == "PE_SELL":   
                accumulator = MAX_BUCKET
                current_state = "BUY"
                final_signals.append(f"🟢 SHEET HINT HIT: PE SELL [STRONG BUY]")
            elif hint_match == "CE_SELL": 
                accumulator = MIN_BUCKET
                current_state = "SELL"
                final_signals.append(f"🔴 SHEET HINT HIT: CE SELL [STRONG SELL]")
        else:
            if p_up >= 0.55:
                accumulator += 1  
            elif p_down >= 0.55:
                accumulator -= 1  
            
            accumulator = max(MIN_BUCKET, min(MAX_BUCKET, accumulator))
            
            if accumulator == MAX_BUCKET:
                current_state = "BUY"
                final_signals.append("🟢 STRONG BUY (Engine Mode [5/5])")
            elif min(prob_downs) and accumulator == MIN_BUCKET:
                current_state = "SELL"
                final_signals.append("🔴 STRONG SELL (Engine Mode [-5/-5])")
            else:
                if current_state == "BUY":
                    final_signals.append(f"🟢 HOLD BUY | Engine (Score: {accumulator})")
                elif current_state == "SELL":
                    final_signals.append(f"🔴 HOLD SELL | Engine (Score: {accumulator})")
                else:
                    final_signals.append(f"⚪ NEUTRAL | Building Conviction (Score: {accumulator})")

        scores_log.append(accumulator)
        calc_raw_weighted = c_val - k_price_val
        raw_weighted_momentum_log.append(calc_raw_weighted)

    # Re-apply structural arrays data logs back onto the dashboard views
    df_predict['d_ML_Signal'] = final_signals
    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 

    # Smooth momentum layer applied with variance filter lock parameter execution P=0.50
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50)

    # Display Configuration Setup
    clean_display_cols = ['a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Weighted_Momentum', 'd_ML_Signal']
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    display_df['Accumulator_Score'] = display_df['Accumulator_Score'].astype(int)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    
    # Re-sorting table array order index matrix
    display_df = display_df.sort_index(ascending=False)
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live 1-Hour Nifty Engine Tracker (Sheet Hints + Kalman 0.50 Logic)")
    st.dataframe(display_df, use_container_width=True, height=750)
