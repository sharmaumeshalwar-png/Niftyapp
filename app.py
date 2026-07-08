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
# AAPKI SHEET KA EXACT DATA MAPPER (Converted to System Timestamps)
# =====================================================================
# 2025 and 2026 dates from image converted strictly to YYYY-MM-DD
hint_sheet_data = {
    # Left Column (2025)
    "2025-07-04 12:15": "CE_SELL", "2025-07-04 15:15": "CE_SELL",
    "2025-07-08 10:15": "CE_SELL", "2025-07-08 13:15": "PE_SELL",
    "2025-07-08 14:15": "CE_SELL", "2025-07-08 15:15": "PE_SELL",
    "2025-07-09 15:15": "CE_SELL", "2025-02-18 10:15": "PE_SELL",
    "2025-08-26 10:15": "CE_SELL", "2025-09-04 10:15": "PE_SELL",
    "2025-09-04 15:15": "CE_SELL", "2025-09-08 10:15": "PE_SELL",
    "2025-09-25 10:15": "CE_SELL", "2025-10-06 10:15": "PE_SELL",
    "2025-11-04 14:15": "CE_SELL", "2025-11-11 15:15": "PE_SELL",
    "2025-11-25 15:15": "CE_SELL", "2025-11-26 10:15": "PE_SELL",
    "2025-12-02 10:15": "CE_SELL", "2025-12-05 12:15": "PE_SELL",
    "2025-12-08 10:15": "CE_SELL", "2025-12-22 10:15": "PE_SELL",
    "2025-12-24 09:15": "CE_SELL",
    # Transition to 2026 (Left Column Bottom)
    "2026-01-01 11:15": "PE_SELL", "2026-01-01 12:15": "CE_SELL",
    
    # Right Column (2025-2026 standard mappings)
    "2025-01-01 14:15": "PE_SELL", "2025-01-01 15:15": "CE_SELL",
    "2025-01-02 10:15": "PE_SELL", "2025-01-06 11:15": "CE_SELL",
    "2025-02-03 10:15": "PE_SELL", "2025-02-05 14:15": "CE_SELL",
    "2025-02-05 15:15": "PE_SELL", "2025-02-06 10:15": "CE_SELL",
    "2025-07-06 12:15": "PE_SELL", "2025-02-06 13:15": "CE_SELL",
    "2025-02-06 14:15": "PE_SELL", "2025-02-13 10:15": "CE_SELL",
    "2025-02-16 15:15": "PE_SELL", "2025-02-19 12:15": "CE_SELL",
    "2025-02-23 10:15": "PE_SELL", "2025-02-23 13:15": "CE_SELL",
    "2025-02-23 15:15": "PE_SELL", "2025-02-24 10:15": "CE_SELL",
    "2025-04-08 10:15": "PE_SELL", "2025-05-12 10:15": "CE_SELL",
    "2025-05-14 12:15": "PE_SELL", "2025-05-29 15:15": "CE_SELL",
    "2025-06-12 11:15": "PE_SELL"
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
    # Download raw Nifty Data
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    
    if len(raw_df) == 0:
        st.error("YFinance API Mismatch or Market Closed. Please refresh dashboard.")
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
        current_time_str = timestamps[i].strftime('%Y-%m-%d %H:%M')

        # Check mapping pattern match in your manual written sheets hints database
        hint_match = hint_sheet_data.get(current_time_str, None)

        if hint_match is not None:
            # Override standard indicators to enforce absolute sheet instructions
            if hint_match == "PE_SELL":   # Bullish Market Bias (Selling Put Options)
                accumulator = MAX_BUCKET
                current_state = "BUY"
                final_signals.append(f"🟢 SHEET HINT HIT: PE SELL [STRONG BUY]")
            elif hint_match == "CE_SELL": # Bearish Market Bias (Selling Call Options)
                accumulator = MIN_BUCKET
                current_state = "SELL"
                final_signals.append(f"🔴 SHEET HINT HIT: CE SELL [STRONG SELL]")
        else:
            # Fallback seamlessly to the mathematical calculation engine when no manual hint matches
            if p_up >= 0.55:
                accumulator += 1  
            elif p_down >= 0.55:
                accumulator -= 1  
            
            accumulator = max(MIN_BUCKET, min(MAX_BUCKET, accumulator))
            
            if accumulator == MAX_BUCKET:
                current_state = "BUY"
                final_signals.append("🟢 STRONG BUY (Engine Mode [5/5])")
            elif accumulator == MIN_BUCKET:
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

    # Smooth momentum layer applied with
