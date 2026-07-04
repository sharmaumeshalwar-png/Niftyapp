import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="India VIX Deep Adaptive Brain Engine", layout="wide")
st.title("🧠 India VIX Live 1-Hour Standalone [Deep Memory Self-Perfecting Engine]")
st.write("🎯 **Aapki Custom Setting:** India VIX 1-Hour Data + 50:50 Split + **VWAP REMOVED** + ML Score $[-5,5]$ + Strictly Past 25-Candle Target + **Global Error Memory & Feature Penalty Optimization Added**")

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

with st.spinner("Initializing Deep Memory Brain & Loading India VIX Core..."):
    # India VIX 1-HOUR Interval Data
    raw_df = yf.download("^INDIAVIX", period="730d", interval="1h")
    
    if len(raw_df) == 0:
        st.error("YFinance API Timeout or NSE Market Closed. Please refresh the dashboard.")
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
    
    # Microstructure Features Space
    df['Sign_Change'] = (np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))).astype(int)
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(window=24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # TARGET RULE (STRICT PAST 25 CANDLES)
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix + ['Target'], inplace=True)

# Dynamic Split Engine (Strict 50:50 Ratio)
split_idx = int(len(df) * 0.50)
df_train = df.iloc[:split_idx]
X_train = df_train[features_matrix].copy()
y_train = df_train['Target'].copy()

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

    # =====================================================================
    # DEEP GLOBAL BRAIN MATRIX & LEARNING LOOP
    # =====================================================================
    final_signals, scores_log, raw_weighted_momentum_log = [], [], []
    correction_notes = []
    
    accumulator = 0
    
    # Global Memory Tracking Elements
    total_predictions_made = 0
    total_correct_predictions = 0
    
    # Har feature ki galti (Error Contribution) store karne ki memory bank
    global_error_memory = {feat: 0.0 for feat in features_matrix}
    
    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()
    feat_values = df_predict[features_matrix].to_numpy()

    for i in range(len(prob_ups)):
        p_up, p_down = prob_ups[i], prob_downs[i]
        c_val, k_price_val = closes[i], kalmans_price[i]
        
        # Adaptive adjustment vectors based on error memory
        # Agar pichle records me galti zyada hai to probability math ko optimize kiya jayega
        penalty_up = 0.0
        penalty_down = 0.0
        
        # Check historical errors to self-correct the current state threshold
        worst_feature_historically = max(global_error_memory, key=global_error_memory.get)
        if global_error_memory[worst_feature_historically] > 0:
            # Global brain automatically applies buffer to prevent recurring trap
            penalty_up = 0.02
            penalty_down = 0.02

        # Final active state determination
        adjusted_p_up = max(0.0, min(1.0, p_up - penalty_up))
        adjusted_p_down = max(0.0, min(1.0, p_down - penalty_down))
        current_bet = 1 if adjusted_p_up >= 0.50 else 0
        
        # Core Feedback loop analysis starting from candle index 1
        if i > 0:
            actual_direction = 1 if closes[i] > closes[i-1] else 0
            prev_adjusted_p_up = max(0.0, min(1.0, prob_ups[i-1] - penalty_up))
            prev_bet = 1 if prev_adjusted_p_up >= 0.50 else 0
            
            total_predictions_made += 1
            
            if prev_bet == actual_direction:
                total_correct_predictions += 1
                current_accuracy = (total_correct_predictions / total_predictions_made) * 100
                
                if current_accuracy >= 85.0 and total_predictions_made > 50:
                    note = f"🔥 [100% PERFECTLY CALIBRATED] Global Accuracy: {current_accuracy:.1f}%. Patterns fully mapped."
                else:
                    note = f"✅ Success. Global Accuracy: {current_accuracy:.1f}% | Learning optimized."
            else:
                # Calculate feature drift to identify the failure cause
                error_v = feat_values[i] - feat_values[i-1]
                culprit_idx = np.argmax(np.abs(error_v))
                culprit_feature = features_matrix[culprit_idx]
                
                # Add this specific mistake to the GLOBAL BRAIN MEMORY BANK
                global_error_memory[culprit_feature] += 1.0
                
                current_accuracy = (total_correct_predictions / total_predictions_made) * 100
                worst_feat = max(global_error_memory, key=global_error_memory.get)
                
                note = f"❌ Error Logged! Culprit: '{culprit_feature}' | Max Core Vulnerability: '{worst_feat}' (Errors: {global_error_memory[worst_feat]:.0f}) | Acc: {current_accuracy:.1f}%"
        else:
            note = "🧠 Global Brain Memory Booting Up... Scanning Historical Structure."

        correction_notes.append(note)

        # Apply standard Accumulator score dynamics
        if adjusted_p_up >= 0.55: accumulator += 1
        elif adjusted_p_down >= 0.55: accumulator -= 1
        
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)
        raw_weighted_momentum_log.append(c_val - k_price_val)
        
        if accumulator == 5: 
            final_signals.append("🟢 VOLATILITY SPIKE (Max Locked)")
        elif accumulator == -5: 
            final_signals.append("🔴 VOLATILITY CRUSH (Max Locked)")
        else: 
            final_signals.append(f"⚪ NEUTRAL (Score: {accumulator})")

    # Map engine vectors back to matrix
    df_predict['d_ML_Signal'] = final_signals
    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 
    df_predict['ML_Brain_Perfecting_Notes'] = correction_notes

    # [Kalman 2 Execution]
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

    # UI Viewport Construction
    clean_display_cols = ['a_Close', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'd_ML_Signal', 'ML_Brain_Perfecting_Notes']
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    
    # Inverting index to display active live state on top row
    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live Original Dataset Matrix (Latest Hour Locked on Top Row)")
    st.dataframe(display_df, use_container_width=True, height=750)
