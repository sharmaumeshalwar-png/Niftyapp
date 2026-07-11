import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC Anti-Flip Laser Engine", layout="wide")
st.title("⚡ BTC-USD Live 1-Hour [Anti-Flip Hysteresis Memory Engine]")
st.write("🎯 **Aapki Custom Setting:** Normalized_Gap + Volatility Cutoff + **Strict Anti-Flip Memory Lock (70% Hysteresis Cutoff)** + `min_samples_leaf=1` + 50:50 Split")

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

with st.spinner("Executing Anti-Flip Filter Engine for BTC-USD..."):
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
    
    # Hard Cutoff Volatility Avg Baseline
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
    model_flow = RandomForestClassifier(n_estimators=150, min_samples_leaf=1, random_state=42)
    model_flow.fit(X_train, y_train)

    # Raw Feature Probability Extraction Before Override
    feat_raw_probs = {}
    for feat in features_matrix:
        feat_model = RandomForestClassifier(n_estimators=150, min_samples_leaf=1, random_state=42)
        feat_model.fit(X_train[[feat]], y_train)
        feat_raw_probs[feat] = feat_model.predict_proba(X_predict[[feat]])[:, 1]

    # Dynamic Volatility Extraction for Predict Window
    vol_current = df.iloc[split_idx:]['Rolling_Volatility'].to_numpy()
    vol_baseline = df.iloc[split_idx:]['Volatility_7Day_Avg'].to_numpy()
    
    # Memory and Logs Setup
    master_scores_log, feat_scores_log, raw_weighted_momentum_log = [], [], []
    prob_up_log, prob_down_log = [], []
    gap_prob_log, vol_prob_log = [], []
    
    raw_probabilities = model_flow.predict_proba(X_predict)
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()

    # --- INITIAL STATE FOR ANTI-FLIP MEMORY ---
    current_state = 0.500  # Default neutral
    master_accumulator = 0

    # -----------------------------------------------------------------
    # ANTI-FLIP HYSTERESIS LOOP EXECUTION
    # -----------------------------------------------------------------
    for i in range(len(raw_probabilities)):
        p_down_raw = raw_probabilities[i, 0]
        p_up_raw = raw_probabilities[i, 1]
        
        feat_gap_raw = feat_raw_probs['Normalized_Gap'][i]
        feat_vol_raw = feat_raw_probs['Rolling_Volatility'][i]
        
        # 1. Volatility Cutoff Filter
        if vol_current[i] < vol_baseline[i]:
            p_up = 0.500
            p_down = 0.500
            p_gap = 0.500
            p_vol = 0.500
            current_state = 0.500 # Range bound wipes current trend memory
        else:
            # 2. BHAI KA POINT: STRICT ANTI-FLIP MEMORY FILTER (Hysteresis)
            if current_state == 0.500:
                # Agar state neutral hai, toh 0.55+ hone par hi naya state banega
                if p_up_raw >= 0.55: current_state = 1.0
                elif p_down_raw >= 0.55: current_state = 0.0
            elif current_state == 1.0:
                # Agar hum bullish trend me locked hain, toh tabhi change hoga jab down probability strictly > 70% ho!
                if p_down_raw >= 0.70: current_state = 0.0
            elif current_state == 0.0:
                # Agar hum bearish trend me locked hain, toh tabhi change hoga jab up probability strictly > 70% ho!
                if p_up_raw >= 0.70: current_state = 1.0

            # Override output probabilities based on locked state memory
            if current_state == 1.0:
                p_up, p_down = p_up_raw, p_down_raw
                p_gap, p_vol = feat_gap_raw, feat_vol_raw
            elif current_state == 0.0:
                p_up, p_down = p_up_raw, p_down_raw
                p_gap, p_vol = feat_gap_raw, feat_vol_raw
            else:
                p_up, p_down = 0.500, 0.500
                p_gap, p_vol = 0.500, 0.500
            
        prob_up_log.append(p_up)
        prob_down_log.append(p_down)
        gap_prob_log.append(p_gap)
        vol_prob_log.append(p_vol)

        # 3. Smooth Accumulator (Will not flip randomly)
        if p_up >= 0.55 and current_state == 1.0: master_accumulator += 1
        elif p_down >= 0.55 and current_state == 0.0: master_accumulator -= 1
        
        master_accumulator = max(-5, min(5, master_accumulator))
        master_scores_log.append(master_accumulator)
        
        # Feature Accumulator
        current_feat_score = 0
        if p_gap >= 0.55: current_feat_score += 1
        elif p_gap <= 0.45: current_feat_score -= 1
        if p_vol >= 0.55: current_feat_score += 1
        elif p_vol <= 0.45: current_feat_score -= 1
        feat_scores_log.append(current_feat_score)
        
        raw_weighted_momentum_log.append(closes[i] - kalmans_price[i])

    # Mapping safe arrays back to presentation layer
    df_predict['Prob_Up'] = prob_up_log
    df_predict['Prob_Down'] = prob_down_log
    df_predict['P_Up_Normalized_Gap'] = gap_prob_log
    df_predict['P_Up_Rolling_Volatility'] = vol_prob_log
    
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
    
    # Top Row Freeze
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Anti-Flip Laser Engine Grid (Hysteresis Memory Lock Enabled)")
    st.dataframe(display_df, use_container_width=True, height=750)
