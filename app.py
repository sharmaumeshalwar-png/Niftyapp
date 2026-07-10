import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC Core Pulse Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Hour [Zero-Leakage Engine 10x]")
st.write("🎯 **Upgraded Setting:** Strictly Causal Processing | 50:50 Split First -> Feature Engineering Second (No Future Contamination)")

# =====================================================================
# MATHEMATICAL ENGINE (Causal Kalman Filter)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0:
        return np.array([])
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

# =====================================================================
# FEATURE ENGINEERING FUNCTION (Strictly Safe - Runs independently on slices)
# =====================================================================
def build_microstructure_features(sliced_df):
    # Copy to avoid setting with copy warning
    df_out = sliced_df.copy()
    
    df_out['a_Close'] = df_out['Close']
    df_out['b_Kalman_Price'] = apply_kalman_filter_custom(df_out['a_Close'].values, initial_p=50.0, q_val=0.001, r_val=0.1)
    df_out['c_Combined'] = df_out['a_Close'] - df_out['b_Kalman_Price']
    
    # 1. Standard Core Imbalances
    df_out['Sign_Change'] = (np.sign(df_out['c_Combined']) != np.sign(df_out['c_Combined'].shift(1))).astype(int)
    df_out['Order_Imbalance'] = (df_out['a_Close'] - df_out['Low']) / (df_out['High'] - df_out['Low'] + 1e-10)
    df_out['Body_Center'] = (df_out['Open'] + df_out['a_Close']) / 2
    df_out['Body_Imbalance'] = (df_out['Body_Center'] - df_out['Low']) / (df_out['High'] - df_out['Low'] + 1e-10)
    df_out['Normalized_Gap'] = df_out['c_Combined'] / (df_out['c_Combined'].rolling(window=24).std() + 1e-10)
    
    # 2. Kinematics (Velocity, Acceleration, Jerk)
    df_out['Flow_Velocity'] = df_out['c_Combined'].diff(1)
    df_out['Acceleration_Force'] = df_out['Flow_Velocity'].diff(1) 
    df_out['Jerk_Force'] = df_out['Acceleration_Force'].diff(1) 
    
    # 3. Micro-Spread & Range Expansion
    hl_range = df_out['High'] - df_out['Low']
    atr_proxy = hl_range.rolling(window=14).mean()
    df_out['Vol_Shock'] = np.where(atr_proxy > 0, hl_range / (atr_proxy + 1e-10), 1.0)
    df_out['Micro_Spread_Proxy'] = (df_out['High'] - df_out['Low']) / (abs(df_out['a_Close'] - df_out['Open']) + 1e-10)
    
    # 4. Range Compression (Parkinson Volatility Proxy)
    df_out['Log_HL'] = np.log(df_out['High'] / (df_out['Low'] + 1e-10))
    df_out['Parkinson_Vol'] = np.sqrt((df_out['Log_HL'] ** 2) / (4 * np.log(2)))
    df_out['Vol_Squeeze_Ratio'] = df_out['Parkinson_Vol'] / (df_out['Parkinson_Vol'].rolling(window=20).mean() + 1e-10)
    
    # 5. Volume-Weighted Price Discovery Speed
    df_out['Candle_Direction'] = np.where(df_out['a_Close'] > df_out['Open'], 1, -1)
    df_out['Volume_Imbalance_Weight'] = df_out['Volume'] * df_out['Candle_Direction'] * df_out['Order_Imbalance']
    df_out['Volume_Imbalance_EMA'] = df_out['Volume_Imbalance_Weight'].ewm(span=5, adjust=False).mean()
    
    # 6. Advanced Efficiency (Fractal Dimension Proxy)
    net_change_10 = (df_out['a_Close'] - df_out['a_Close'].shift(10)).abs()
    sum_changes_10 = df_out['a_Close'].diff().abs().rolling(window=10).sum()
    df_out['Noise_Ratio_10'] = net_change_10 / (sum_changes_10 + 1e-10)
    
    net_change_5 = (df_out['a_Close'] - df_out['a_Close'].shift(5)).abs()
    sum_changes_5 = df_out['a_Close'].diff().abs().rolling(window=5).sum()
    df_out['Noise_Ratio_5'] = net_change_5 / (sum_changes_5 + 1e-10)

    # 7. Mean-Reversion Force
    rolling_std = df_out['a_Close'].rolling(window=20).std()
    df_out['Band_Distance'] = (df_out['a_Close'] - df_out['a_Close'].rolling(window=20).mean()) / (rolling_std + 1e-10)
    
    # Target Mapping
    df_out['State_Direction'] = np.where(df_out['c_Combined'] > 0, 1, 0)
    
    return df_out

with st.spinner("Executing Uncontaminated Live Data Pipeline..."):
    current_time = datetime.now()
    start_date = current_time - timedelta(days=720) 
    end_date = current_time + timedelta(days=1) 
    
    raw_df = yf.download("BTC-USD", start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1h", auto_adjust=True)
    
    if raw_df.empty:
        st.error("YFinance Limit Hit. Refresh.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df.index = pd.to_datetime(df.index)

    # =====================================================================
    # 🛡️ THE GOLDEN SHIELD: SPLIT FIRST, CALCULATION SECOND
    # =====================================================================
    split_idx = int(len(df) * 0.50)
    
    # Pure raw splits with zero indicator values inherited
    raw_train = df.iloc[:split_idx].copy()
    raw_predict = df.iloc[split_idx:].copy()
    
    # Processing both universes in total math isolation
    df_train = build_microstructure_features(raw_train)
    df_predict = build_microstructure_features(raw_predict)

    # 14x Hyper Features Matrix
    features_matrix = [
        'c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity',
        'Acceleration_Force', 'Jerk_Force', 'Vol_Shock', 'Micro_Spread_Proxy', 
        'Vol_Squeeze_Ratio', 'Volume_Imbalance_EMA', 'Noise_Ratio_10', 'Noise_Ratio_5', 'Band_Distance'
    ]
    
    df_train.dropna(subset=features_matrix + ['State_Direction'], inplace=True)
    df_predict.dropna(subset=features_matrix + ['State_Direction'], inplace=True)

    X_train = df_train[features_matrix]
    y_train = df_train['State_Direction']
    X_predict = df_predict[features_matrix]

if len(X_predict) == 0:
    st.error("Data tracking frame mismatch.")
else:
    # Anti-Overfit Forest Setup
    model_flow = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=7, max_features='log2', random_state=42, n_jobs=-1)
    model_flow.fit(X_train, y_train)

    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # Signal Accumulator Flow
    scores_log = []
    accumulator = 0
    current_signal = 0  
    consecutive_counter = 0
    required_confirmations = 2 

    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()

    for i in range(len(prob_ups)):
        p_up, p_down = prob_ups[i], prob_downs[i]
        potential_signal = 1 if p_up >= 0.53 else (-1 if p_down >= 0.53 else current_signal)

        if potential_signal != current_signal and current_signal != 0:
            consecutive_counter += 1
            if consecutive_counter >= required_confirmations:
                current_signal = potential_signal
                consecutive_counter = 0
        else:
            current_signal = potential_signal
            consecutive_counter = 0

        if current_signal == 1: accumulator += 1
        elif current_signal == -1: accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)

    df_predict['Accumulator_Score'] = scores_log  
    
    # Causal Post-Smoother for UI
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['c_Combined'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # Momentum MACD Layout
    fast_ema = df_predict['Weighted_Momentum'].ewm(span=10, adjust=False).mean()
    slow_ema = df_predict['Weighted_Momentum'].ewm(span=50, adjust=False).mean()
    df_predict['Mom_MACD_Line'] = fast_ema - slow_ema
    df_predict['Mom_MACD_Signal'] = df_predict['Mom_MACD_Line'].ewm(span=200, adjust=False).mean()
    df_predict['Mom_MACD_Hist'] = df_predict['Mom_MACD_Line'] - df_predict['Mom_MACD_Signal']

    # Adaptive Structural Actions
    df_predict['Price_Delta'] = df_predict['a_Close'].diff(1)
    df_predict['Signal_Flip'] = (df_predict['State_Direction'] != df_predict['State_Direction'].shift(1)).astype(int)
    market_noise_threshold = df_predict['Price_Delta'].rolling(window=24).std()
    df_predict['Trap_Flip'] = np.where((df_predict['Signal_Flip'] == 1) & (df_predict['Price_Delta'].abs() > market_noise_threshold), 1, 0)
    
    adaptive_actions = []
    for idx, row in df_predict.iterrows():
        if row['Trap_Flip'] == 1:
            if row['Body_Imbalance'] > 0.80 or row['Body_Imbalance'] < 0.20:
                adaptive_actions.append("🛡️ Body Imbalance Zone: Locked")
            elif row['Order_Imbalance'] > 0.85 or row['Order_Imbalance'] < 0.15:
                adaptive_actions.append("⚠️ Order Flow Vacuum: Risk")
            else:
                adaptive_actions.append("📈 Expansion Risk Management")
        else:
            adaptive_actions.append("✅ Structure Balanced" if 0.25 <= row['Body_Imbalance'] <= 0.75 else "👀 Liquidity Shift Building")
            
    df_predict['Optimized_Setting_Action'] = adaptive_actions

    # Dynamic Energy Pulse
    df_predict['Feature_Energy'] = df_predict['Order_Imbalance'].diff().abs() + df_predict['Flow_Velocity'].diff().abs()
    energy_threshold = df_predict['Feature_Energy'].rolling(window=20).mean() + (1.5 * df_predict['Feature_Energy'].rolling(window=20).std())
    df_predict['Micro_Momentum_Pulse'] = np.where(df_predict['Feature_Energy'] > energy_threshold, "⚡ PULSE ACTIVE", "⚪ Stable Noise")

    # Final Display Inversion
    clean_display_cols = [
        'Open', 'a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Weighted_Momentum', 
        'Mom_MACD_Line', 'Mom_MACD_Signal', 'Mom_MACD_Hist', 'Micro_Momentum_Pulse', 'Optimized_Setting_Action'
    ]
    display_df = df_predict[clean_display_cols].copy()
    
    round_cols = ['Open', 'a_Close', 'b_Kalman_Price', 'Weighted_Momentum', 'Mom_MACD_Line', 'Mom_MACD_Signal', 'Mom_MACD_Hist']
    display_df[round_cols] = display_df[round_cols].round(2)
    display_df[['Prob_Up', 'Prob_Down']] = display_df[['Prob_Up', 'Prob_Down']].round(3)
    
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader("📋 Uncontaminated Microstructure Engine Live Matrix")
    st.dataframe(display_df, use_container_width=True, height=750)
