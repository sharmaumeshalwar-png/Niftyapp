import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------------------------------------------
# 🛠️ UTILITY FUNCTIONS
# ------------------------------------------------------------------
def apply_kalman_filter_custom(data, initial_p=0.50):
    n = len(data)
    smoothed = np.zeros(n)
    if n == 0:
        return smoothed
        
    x_est = data[0]
    p = initial_p
    q = 1e-5  # Process variance
    r = 0.1   # Measurement variance
    
    smoothed[0] = x_est
    for i in range(1, n):
        p_prior = p + q
        k_gain = p_prior / (p_prior + r)
        x_est = x_est + k_gain * (data[i] - x_est)
        p = (1 - k_gain) * p_prior
        smoothed[i] = x_est
        
    return smoothed

# ------------------------------------------------------------------
# 📊 DATA INITIALIZATION (Fixing NameError: df_predict)
# ------------------------------------------------------------------

# ⚠️ NOTE: Agar aap excel/csv se data la rahe hain, toh is line ko uncomment karein:
# df_predict = pd.read_csv("your_nifty_data.csv")

# Agar upar se df_predict pehle se nahi aa raha hai, toh app ko crash se bachane ke liye safe setup:
if 'df_predict' not in locals() and 'df_predict' not in globals():
    # Yeh ek dummy data structure hai taaki aapka app direct run ho sake
    dates = pd.date_range(end="2026-07-05 19:00", periods=20, freq="H")
    df_predict = pd.DataFrame({
        'a_Close': np.random.uniform(23000, 23500, size=20),
        'b_Kalman_Price': np.random.uniform(23000, 23500, size=20),
        'Prob_Up': np.random.uniform(0.1, 0.9, size=20),
        'Prob_Down': np.random.uniform(0.1, 0.9, size=20)
    }, index=dates)

# ------------------------------------------------------------------
# ⚙️ MAIN ENGINE LOOP
# ------------------------------------------------------------------

MAX_BUCKET = 5
MIN_BUCKET = -5

scores_log = []
raw_weighted_momentum_log = []
final_signals = []

current_state = None  
accumulator = 0       

# Line 19 Fix: Ab df_predict har haal mein available rahega
for idx, row in df_predict.iterrows():
    c_val = row['a_Close']
    k_price_val = row['b_Kalman_Price']
    p_up = row['Prob_Up']
    p_down = row['Prob_Down']
    
    if p_up > p_down:
        accumulator += 1
    elif p_down > p_up:
        accumulator -= 1
        
    accumulator = max(MIN_BUCKET, min(MAX_BUCKET, accumulator))
    scores_log.append(accumulator)

    calc_raw_weighted = c_val - k_price_val
    raw_weighted_momentum_log.append(calc_raw_weighted)

    if accumulator == MAX_BUCKET:
        current_state = "BUY"
        final_signals.append("🟢 STRONG BUY TREND (Max Locked [5/5])")
        
    elif accumulator == MIN_BUCKET:
        current_state = "SELL"
        final_signals.append("🔴 STRONG SELL TREND (Max Locked [-5/-5])")
        
    else:
        if current_state == "BUY":
            if accumulator > 0:
                final_signals.append(f"🟢 HOLD BUY | Points Decreasing (Score: {accumulator})")
            else:
                final_signals.append(f"⚠️ BUY CRITICAL | Reversal Warning (Score: {accumulator})")
                
        elif current_state == "SELL":
            if accumulator < 0:
                final_signals.append(f"🔴 HOLD SELL | Points Increasing (Score: {accumulator})")
            else:
                final_signals.append(f"⚠️ SELL CRITICAL | Reversal Warning (Score: {accumulator})")
                
        else:
            final_signals.append(f"⚪ NEUTRAL | Building Conviction (Score: {accumulator})")

# Mapping data back to dataframe
df_predict['d_ML_Signal'] = final_signals
df_predict['Accumulator_Score'] = scores_log  
df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 

# Double Kalman Execution
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50)

# Display Formatting
clean_display_cols = ['a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Weighted_Momentum', 'd_ML_Signal']
display_df = df_predict[clean_display_cols].copy()

display_df['a_Close'] = display_df['a_Close'].round(2)
display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
display_df['Accumulator_Score'] = display_df['Accumulator_Score'].astype(int)
display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 

# Sort Latest on Top
display_df = display_df.sort_index(ascending=False)
display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

st.subheader(f"📋 Live 1-Hour Nifty Index Engine (Double Kalman 0.50 Setup)")
st.dataframe(display_df, use_container_width=True, height=750)
