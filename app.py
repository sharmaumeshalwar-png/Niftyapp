import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC Hard-Cutoff Laser Engine", layout="wide")
st.title("⚡ BTC-USD Live 1-Hour [Strict Hard-Cutoff Volatility Filter]")
st.write("🎯 **Aapki Custom Setting:** Normalized_Gap + **Strict Out-of-Model Hard Cutoff** (Volatility < 7-Day Avg = Forced 0.50 Neutral) + `min_samples_leaf=1` + 50:50 Split")

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

with st.spinner("Executing Strict Hard-Cutoff Engine for BTC-USD..."):
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
    
    # Core Mathematical Metrics
    df['Rolling_Volatility'] = df['c_Combined'].rolling(window=24).std()
    df['Normalized_Gap'] = df['c_Combined'] / (df['Rolling_Volatility'] + 1e-10)
    
    # --- HARD CUTOFF BENCHMARK ---
    # Pichle 7 dino (168 ghante) ka average standard deviation baseline
    df['Volatility_7Day_Avg'] = df['Rolling_Volatility'].rolling(window=168).mean()
    
    # Target Direction State
    df['State_Direction'] = np.where(df['c_Combined'] > 0, 1, 0)
    
    # Clean up matrix
    features_matrix = ['Normalized_Gap', 'Rolling_Volatility']
    df.dropna(subset=features_matrix + ['State_Direction', 'Volatility_7Day_Avg'], inplace=True)

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

    # Individual Feature Breakdown Engine
    feat_probs = {}
    for feat in features_matrix:
        feat_model = RandomForestClassifier(n_estimators=150, min_samples_leaf=1, random_state=42)
        feat_model.fit(X_train[[feat]], y_train)
        col_name = f'P_Up_{feat}'
        df_predict[col_name] = feat_model.predict_proba(X_predict[[feat]])[:, 1]
        feat_probs[col_name] = df_predict[col_name].to_numpy()

    # Dynamic Volatility Extraction for Predict Window
    vol_current = df.iloc[split_idx:]['Rolling_Volatility'].to_numpy()
    vol_baseline = df.iloc[split_idx:]['Volatility_7Day_Avg'].to_numpy()
    
    # Live Accumulators & Raw Logs
    master_scores_log, feat_scores_log, raw_weighted_momentum_log = [], [], []
    prob_up_log, prob_down_log = [], []
    master_accumulator = 0
    
    raw_probabilities = model_flow.predict_proba(X_predict)
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()

    # -----------------------------------------------------------------
    # THE HARD CUTOFF RULE EXECUTION LOOP
    # -----------------------------------------------------------------
    for i in range(len(raw_probabilities)):
        p_down_raw = raw_probabilities[i, 0]
        p_up_raw = raw_probabilities[i, 1]
        
        # CONDITION: Agar current standard deviation weekly average se kam hai -> RESET EVERYTHING TO NEUTRAL
        if vol_current[i] < vol_baseline[i]:
            p_up = 0.500
            p_down = 0.500
            # Columns breakdown ko bhi override kar do taaki fake indications na dikhein
            feat_probs['P_Up_Normalized_Gap'][i] = 0.500
            feat_probs['P_Up_Rolling_Volatility'][i] = 0.500
        else:
            p_up = p_up_raw
            p_down = p_down_raw
            
        prob_up_log.append(p_up)
        prob_down_log.append(p_down)

        # Master Accumulator Logic [-5, 5] (Ab rangebound me ye freeze rahega)
        if p_up >= 0.55: master_accumulator += 1
        elif p_down >= 0.55: master_accumulator -= 1
        master_accumulator = max(-5, min(5, master_accumulator))
        master_scores_log.append(master_accumulator)
        
        # Feature Accumulator Logic [-2, 2]
        current_feat_score = 0
        for feat in features_matrix:
            f_prob = feat_probs[f'P_Up_{feat}'][i]
            if f_prob >= 0.55: current_feat_score += 1
            elif f_prob <= 0.45: current_feat_score -= 1
        feat_scores_log.append(current_feat_score)
        
        raw_weighted_momentum_log.append(closes[i] - kalmans_price[i])

    # Assigning Clean Logs
    df_predict['Prob_Up'] = prob_up_log
    df_predict['Prob_Down'] = prob_down_log
    df_predict['P_Up_Normalized_Gap'] = feat_probs['P_Up_Normalized_Gap']
    df_predict['P_Up_Rolling_Volatility'] = feat_probs['P_Up_Rolling_Volatility']
    
    df_predict['Accumulator_Score'] = master_scores_log  
    df_predict['Feature_Accumulator'] = feat_scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # Formatting UI Structure
    clean_display_cols = [
        'a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 
        'P_Up_Normalized_Gap', 'P_Up_Rolling_Volatility',
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

    st.subheader(f"📋 Laser Engine Matrix with Strict 7-Day Volatility Cutoff Filter")
    st.dataframe(display_df, use_container_width=True, height=750)
