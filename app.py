import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import pytz

# Page Configuration
st.set_page_config(page_title="BTC Shock Pulse Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Hour [Umesh Strict Jump Filter]")
st.write("🎯 **Strict Jump Condition:** Trigger ONLY when probability shifts from `< 0.10` directly to `> 0.88`. Normal candles strictly filtered out.")

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

with st.spinner("Scanning Matrix for Strict 0.10 -> 0.88 Probability Jumps..."):
    IST = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(IST)
    
    start_date = current_time - timedelta(days=720) 
    end_date = current_time + timedelta(days=1) 
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    raw_df = yf.download("BTC-USD", start=start_str, end=end_str, interval="1h")
    
    if len(raw_df) == 0:
        st.error(f"YFinance Connection Error. Please refresh.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df.index = pd.to_datetime(df.index)

    # Base Matrix Definition
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0, q_val=0.001, r_val=0.1)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Microstructure Features Space
    df['Sign_Change'] = (np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))).astype(int)
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(window=24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    df['State_Direction'] = np.where(df['c_Combined'] > 0, 1, 0)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix + ['State_Direction'], inplace=True)

# Dynamic Split Engine (Strict 50:50 Ratio)
split_idx = int(len(df) * 0.50)
df_train = df.iloc[:split_idx]
X_train = df_train[features_matrix].copy()
y_train = df_train['State_Direction'].copy()

df_predict = df.iloc[split_idx:].copy()
X_predict = df_predict[features_matrix].copy()

if len(X_predict) == 0:
    st.error("Prediction matrix error.")
else:
    model_flow = RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=1, random_state=42)
    model_flow.fit(X_train, y_train)

    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # Live Accumulators & Raw Logs
    scores_log, raw_weighted_momentum_log = [], []
    accumulator = 0
    
    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()

    for i in range(len(prob_ups)):
        p_up, p_down, c_val, k_price_val = prob_ups[i], prob_downs[i], closes[i], kalmans_price[i]
        if p_up >= 0.55: accumulator += 1
        elif p_down >= 0.55: accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)
        raw_weighted_momentum_log.append(c_val - k_price_val)

    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # -----------------------------------------------------------------
    # 🎯 STRIKT 0.10 -> 0.88 JUMP LOGIC (Umesh Professional Engine)
    # -----------------------------------------------------------------
    strict_shock_logs = []
    current_position = None  # Holds: 'SHOCK_BULL' or 'SHOCK_BEAR'
    
    # Pre-calculated 3.5x SD Buffer for 100% Zero Monthly Strike Expiry Check
    safe_buffer_dist = int(df_predict['a_Close'].rolling(window=24).std().mean() * 3.5)
    
    # We loop safely from index 1 to avoid shift data leakage errors
    for i in range(len(df_predict)):
        if i == 0:
            strict_shock_logs.append("-")
            continue
            
        current_open = df_predict['Open'].iloc[i]
        strike_base = round(current_open / 500) * 500
        
        # Current probabilities
        p_up_curr = df_predict['Prob_Up'].iloc[i]
        p_down_curr = df_predict['Prob_Down'].iloc[i]
        
        # Previous candle probabilities (The Base Check)
        p_up_prev = df_predict['Prob_Up'].iloc[i-1]
        p_down_prev = df_predict['Prob_Down'].iloc[i-1]
        
        # CONDITION 1: Bullish Shock Jump (< 0.10 to > 0.88)
        is_bull_shock = (p_up_prev < 0.10) and (p_up_curr > 0.88)
        
        # CONDITION 2: Bearish Shock Jump (< 0.10 to > 0.88)
        is_bear_shock = (p_down_prev < 0.10) and (p_down_curr > 0.88)
        
        if is_bull_shock and current_position != 'SHOCK_BULL':
            current_position = 'SHOCK_BULL'
            safe_put_strike = int(strike_base - safe_buffer_dist)
            strict_shock_logs.append(f"🟢 SPIKE: SELL PUT {safe_put_strike} 📈")
            
        elif is_bear_shock and current_position != 'SHOCK_BEAR':
            current_position = 'SHOCK_BEAR'
            safe_call_strike = int(strike_base + safe_buffer_dist)
            strict_shock_logs.append(f"🔴 CRASH: SELL CALL {safe_call_strike} 📉")
            
        else:
            # Continuously trailing until the vice-versa exact jump occurs
            if current_position == 'SHOCK_BULL':
                strict_shock_logs.append("⏳ Holding PUT Strike")
            elif current_position == 'SHOCK_BEAR':
                strict_shock_logs.append("⏳ Holding CALL Strike")
            else:
                strict_shock_logs.append("-")

    df_predict['Umesh_Shock_Filter_Log'] = strict_shock_logs

    # Formatting UI Structure
    clean_display_cols = ['Open', 'a_Close', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Umesh_Shock_Filter_Log']
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['Open'] = display_df['Open'].round(2)
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    
    # Inverting via index flip to freeze the latest active hour on Top Row
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live BTC-USD Strict Shock Matrix (Latest Hour Locked on Top Row)")
    st.dataframe(display_df, use_container_width=True, height=750)
