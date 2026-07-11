import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC Raw Momentum Engine", layout="wide")
st.title("⚡ BTC-USD Live 1-Hour [Original Raw Momentum Engine]")
st.write("🎯 **Aapki Original Setting:** Pure Normalized_Gap Only + Absolute `min_samples_leaf=1` + Dual Accumulator + Strict 50:50 Train-Predict Split")

# =====================================================================
# MATHEMATICAL ENGINE (Original Kalman Filter Function)
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

with st.spinner("Fetching Live Market Data for BTC-USD..."):
    current_time = datetime.now()
    start_date = current_time - timedelta(days=720) 
    end_date = current_time + timedelta(days=1) 
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    raw_df = yf.download("BTC-USD", start=start_str, end=end_str, interval="1h")
    
    if len(raw_df) == 0:
        st.error(f"YFinance API Limit Error. Please refresh.")
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
    
    # Pure Original Normalized Gap Feature
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(window=24).std() + 1e-10)
    
    # Original Target Direction State
    df['State_Direction'] = np.where(df['c_Combined'] > 0, 1, 0)
    
    # Feature Matrix Setup
    features_matrix = ['Normalized_Gap']
    df.dropna(subset=features_matrix + ['State_Direction'], inplace=True)

# Dynamic Split Engine (Original Strict 50:50 Ratio)
split_idx = int(len(df) * 0.50)
df_train = df.iloc[:split_idx]
X_train = df_train[features_matrix].copy()
y_train = df_train['State_Direction'].copy()

df_predict = df.iloc[split_idx:].copy()
X_predict = df_predict[features_matrix].copy()

if len(X_predict) == 0:
    st.error("Prediction matrix error.")
else:
    # --- ORIGINAL ABSOLUTE LEAF-1 MODEL CALIBRATION ---
    model_flow = RandomForestClassifier(
        n_estimators=150, 
        min_samples_leaf=1, 
        random_state=42
    )
    model_flow.fit(X_train, y_train)

    # Master Probabilities Output
    probabilities = model_flow.predict_proba(X_predict)
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # Feature Breakdown Column Generation
    feat_probs = {}
    for feat in features_matrix:
        feat_model = RandomForestClassifier(n_estimators=150, min_samples_leaf=1, random_state=42)
        feat_model.fit(X_train[[feat]], y_train)
        col_name = f'P_Up_{feat}'
        df_predict[col_name] = feat_model.predict_proba(X_predict[[feat]])[:, 1]
        feat_probs[col_name] = df_predict[col_name].to_numpy()

    # Original Accumulators Flow Loop
    master_scores_log, feat_scores_log, raw_weighted_momentum_log = [], [], []
    master_accumulator = 0
    
    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()

    for i in range(len(prob_ups)):
        # Master Accumulator Loop [-5, 5]
        p_up, p_down = prob_ups[i], prob_downs[i]
        if p_up >= 0.55: master_accumulator += 1
        elif p_down >= 0.55: master_accumulator -= 1
        master_accumulator = max(-5, min(5, master_accumulator))
        master_scores_log.append(master_accumulator)
        
        # Feature Accumulator Loop [-1, 1] (Kyunki ek hi core feature hai)
        current_feat_score = 0
        for feat in features_matrix:
            f_prob = feat_probs[f'P_Up_{feat}'][i]
            if f_prob >= 0.55: current_feat_score += 1
            elif f_prob <= 0.45: current_feat_score -= 1
        feat_scores_log.append(current_feat_score)
        
        raw_weighted_momentum_log.append(closes[i] - kalmans_price[i])

    df_predict['Accumulator_Score'] = master_scores_log  
    df_predict['Feature_Accumulator'] = feat_scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # Formatting UI Structure
    clean_display_cols = [
        'a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 
        'P_Up_Normalized_Gap', 'Accumulator_Score', 'Feature_Accumulator', 'Weighted_Momentum'
    ]
    
    display_df = df_predict[clean_display_cols].copy()
    
    # Precision rounding
    for col in display_df.columns:
        if 'Prob_' in col or 'P_Up_' in col:
            display_df[col] = display_df[col].round(3)
        elif col not in ['Accumulator_Score', 'Feature_Accumulator']:
            display_df[col] = display_df[col].round(2)
    
    # Original Top Row Freeze Sequence
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Original Raw Momentum Engine Matrix (Latest Hour Locked on Top)")
    st.dataframe(display_df, use_container_width=True, height=750)
