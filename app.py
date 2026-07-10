import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC Core Pulse Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Hour [Core Pulse Engine]")
st.write("🎯 **Aapki Custom Setting:** Strictly Only BTC 1-Hour Data + Strict 2-Year Range + 50:50 Split + **Micro-Momentum Pulse Detector** + Momentum MACD (10,50,200) + Latest Active Candle Locked on Top")

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
    filtered_values = np.empty(len(data_array))
    for i, z in enumerate(data_array):
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values[i] = x
    return filtered_values

with st.spinner("Executing Strict Live BTC Data Fetch & Pulse Scanner..."):
    current_time = datetime.now()
    start_date = current_time - timedelta(days=720) 
    end_date = current_time + timedelta(days=1) 
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    raw_df = yf.download("BTC-USD", start=start_str, end=end_str, interval="1h", auto_adjust=True)
    
    if raw_df.empty:
        st.error(f"YFinance API Limit Error for BTC. Please refresh.")
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
    
    # Clean Binary State Definition
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
    model_flow = RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=1, random_state=42, n_jobs=-1)
    model_flow.fit(X_train, y_train)

    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # Live Accumulators & Raw Logs
    scores_log = []
    accumulator = 0
    
    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()

    for i in range(len(prob_ups)):
        p_up, p_down = prob_ups[i], prob_downs[i]
        if p_up >= 0.55: accumulator += 1
        elif p_down >= 0.55: accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)

    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = closes - kalmans_price
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # =====================================================================
    # 📈 MOMENTUM MACD ENGINE (10, 50, 200)
    # =====================================================================
    fast_ema = df_predict['Weighted_Momentum'].ewm(span=10, adjust=False).mean()
    slow_ema = df_predict['Weighted_Momentum'].ewm(span=50, adjust=False).mean()
    
    df_predict['Mom_MACD_Line'] = fast_ema - slow_ema
    df_predict['Mom_MACD_Signal'] = df_predict['Mom_MACD_Line'].ewm(span=200, adjust=False).mean()
    df_predict['Mom_MACD_Hist'] = df_predict['Mom_MACD_Line'] - df_predict['Mom_MACD_Signal']

    # Microstructure Volatility Pulse Detector
    df_predict['Feature_Energy'] = df_predict['Order_Imbalance'].diff().abs() + df_predict['Flow_Velocity'].diff().abs()
    energy_threshold = df_predict['Feature_Energy'].rolling(window=20).mean() + (1.5 * df_predict['Feature_Energy'].rolling(window=20).std())
    df_predict['Micro_Momentum_Pulse'] = np.where(df_predict['Feature_Energy'] > energy_threshold, "⚡ PULSE ACTIVE", "⚪ Stable Noise")

    # =====================================================================
    # 🧠 BALANCED TARGET ENGINE: IMBLANCE & WHIPSIP AUTO-OPTIMIZATION
    # =====================================================================
    df_predict['Price_Delta'] = df_predict['a_Close'].diff(1)
    df_predict['Signal_Flip'] = (df_predict['State_Direction'] != df_predict['State_Direction'].shift(1)).astype(int)
    market_noise_threshold = df_predict['Price_Delta'].rolling(window=24).std()
    df_predict['Trap_Flip'] = np.where((df_predict['Signal_Flip'] == 1) & (df_predict['Price_Delta'].abs() > market_noise_threshold), 1, 0)
    
    adaptive_actions = []
    for idx, row in df_predict.iterrows():
        if row['Trap_Flip'] == 1:
            if row['Body_Imbalance'] > 0.80 or row['Body_Imbalance'] < 0.20:
                adaptive_actions.append("🛡️ Body Imbalance Zone: Reject Flip, Lock Previous Signal Line")
            elif row['Order_Imbalance'] > 0.85 or row['Order_Imbalance'] < 0.15:
                adaptive_actions.append("⚠️ Order Flow Vacuum: Increase Model Prob Trigger to > 0.65")
            elif abs(row['Normalized_Gap']) > 2.0:
                adaptive_actions.append("🔧 Kalman Variance Calibration: Set R=0.50 to absorb gap")
            else:
                adaptive_actions.append("📈 Expansion Risk: Require Multi-Candle Confirmation")
        else:
            if row['Body_Imbalance'] > 0.75 or row['Body_Imbalance'] < 0.25:
                adaptive_actions.append("👀 Liquidity Shift Building (Body skewed)")
            else:
                adaptive_actions.append("✅ Structure Balanced")
            
    df_predict['Optimized_Setting_Action'] = adaptive_actions

    # Formatting UI Structure (Probabilities Added Back)
    clean_display_cols = [
        'Open', 'a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Weighted_Momentum', 
        'Mom_MACD_Line', 'Mom_MACD_Signal', 'Mom_MACD_Hist', 
        'Micro_Momentum_Pulse', 'Optimized_Setting_Action'
    ]
    display_df = df_predict[clean_display_cols].copy()
    
    round_two = ['Open', 'a_Close', 'b_Kalman_Price', 'Weighted_Momentum', 'Mom_MACD_Line', 'Mom_MACD_Signal', 'Mom_MACD_Hist']
    display_df[round_two] = display_df[round_two].round(2)
    display_df[['Prob_Up', 'Prob_Down']] = display_df[['Prob_Up', 'Prob_Down']].round(3)
    
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live BTC-USD Dataset Matrix (Imbalance Zone Core Scanner Enabled)")
    
    def highlight_matrix(val):
        if "🛡️" in str(val) or "⚠️" in str(val):
            return 'background-color: rgba(255, 75, 255, 0.15); color: #ff4bff; font-weight: bold;'
        elif "🔧" in str(val) or "👀" in str(val):
            return 'background-color: rgba(255, 165, 0, 0.15); color: #ffaa00; font-weight: bold;'
        elif "⚡ PULSE ACTIVE" in str(val):
            return 'background-color: rgba(255, 75, 75, 0.15); color: #ff4b4b; font-weight: bold;'
        return ''

    st.dataframe(
        display_df.style.map(highlight_matrix, subset=['Micro_Momentum_Pulse', 'Optimized_Setting_Action']),
        use_container_width=True, 
        height=750
    )
