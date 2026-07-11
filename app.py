import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC Absolute Leaf-1 3-Feature Engine", layout="wide")
st.title("⚡ BTC-USD Live 1-Hour [3-Feature Microstructure Decoder]")
st.write("🎯 **Aapki Custom Setting:** Normalized_Gap REMOVED + Reverted to Extreme `min_samples_leaf=1` + 3 Column Feature Breakdowns + **New Feature Accumulator [-3,3]** + 50:50 Split + Latest Active Candle Locked on Top")

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

with st.spinner("Executing Strict Live Data Fetch for BTC-USD..."):
    # HARDCODED DATE CALCULATOR: Adding +1 Day Buffer to catch TODAY's live candles
    current_time = datetime.now()
    start_date = current_time - timedelta(days=720) 
    end_date = current_time + timedelta(days=1) 
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Fetch using exact BTC-USD ticker
    raw_df = yf.download("BTC-USD", start=start_str, end=end_str, interval="1h")
    
    if len(raw_df) == 0:
        st.error(f"YFinance API Limit Error. Please refresh.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df.index = pd.to_datetime(df.index)

    # Base Matrix Definition (Price Kalman 1 Active)
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0, q_val=0.001, r_val=0.1)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Microstructure Features Space (Normalized_Gap is completely omitted)
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Clean Binary State Definition based on Microstructure direction
    df['State_Direction'] = np.where(df['c_Combined'] > 0, 1, 0)
    
    # Strictly 3 Features left now
    features_matrix = ['Order_Imbalance', 'Body_Imbalance', 'Flow_Velocity']
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
    # --- ABSOLUTE LEAF-1 MODEL CALIBRATION ---
    model_flow = RandomForestClassifier(
        n_estimators=150, 
        min_samples_leaf=1, 
        random_state=42
    )
    model_flow.fit(X_train, y_train)

    # Master Probabilities based on 3-features combination
    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # -----------------------------------------------------------------
    # THE 3 COLUMN FEATURE DECODER ENGINE
    # -----------------------------------------------------------------
    target_features = ['Order_Imbalance', 'Body_Imbalance', 'Flow_Velocity']
    
    # Temporarily store individual probability columns for array operation
    feat_probs = {}
    for feat in target_features:
        feat_model = RandomForestClassifier(n_estimators=150, min_samples_leaf=1, random_state=42)
        feat_model.fit(X_train[[feat]], y_train)
        col_name = f'P_Up_{feat}'
        df_predict[col_name] = feat_model.predict_proba(X_predict[[feat]])[:, 1]
        feat_probs[col_name] = df_predict[col_name].to_numpy()

    # Live Accumulators & Raw Logs
    master_scores_log, feat_scores_log, raw_weighted_momentum_log = [], [], []
    master_accumulator = 0
    
    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()

    # Loop to calculate Master Accumulator AND the new 3-Feature Accumulator
    for i in range(len(prob_ups)):
        # 1. Master Accumulator Logic [-5, 5]
        p_up, p_down = prob_ups[i], prob_downs[i]
        if p_up >= 0.55: master_accumulator += 1
        elif p_down >= 0.55: master_accumulator -= 1
        master_accumulator = max(-5, min(5, master_accumulator))
        master_scores_log.append(master_accumulator)
        
        # 2. The 3-Feature Standalone Accumulator Logic [-3, 3] per row
        current_feat_score = 0
        for feat in target_features:
            f_prob = feat_probs[f'P_Up_{feat}'][i]
            if f_prob >= 0.55:
                current_feat_score += 1
            elif f_prob <= 0.45:
                current_feat_score -= 1
        feat_scores_log.append(current_feat_score)
        
        raw_weighted_momentum_log.append(closes[i] - kalmans_price[i])

    df_predict['Accumulator_Score'] = master_scores_log  
    df_predict['Feature_Accumulator'] = feat_scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # Formatting UI Structure with both accumulators
    clean_display_cols = [
        'a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 
        'P_Up_Order_Imbalance', 'P_Up_Body_Imbalance', 'P_Up_Flow_Velocity',
        'Accumulator_Score', 'Feature_Accumulator', 'Weighted_Momentum'
    ]
    
    display_df = df_predict[clean_display_cols].copy()
    
    # Precision rounding
    for col in display_df.columns:
        if 'Prob_' in col or 'P_Up_' in col:
            display_df[col] = display_df[col].round(3)
        elif col not in ['Accumulator_Score', 'Feature_Accumulator']:
            display_df[col] = display_df[col].round(2)
    
    # Inverting via index flip to freeze the latest active hour on Top Row
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live Cleaned 3-Feature Matrix & Dual Accumulator (Latest Hour Locked on Top)")
    st.dataframe(display_df, use_container_width=True, height=750)
