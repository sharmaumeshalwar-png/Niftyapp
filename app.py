import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------------------------------------------
# 🛠️ UTILITY FUNCTIONS (For Smooth Copy-Paste Execution)
# ------------------------------------------------------------------

def apply_kalman_filter_custom(data, initial_p=0.50):
    """
    Custom Kalman Filter Implementation with Initial P parameter.
    Smooths out the Raw Weighted Momentum values.
    """
    n = len(data)
    smoothed = np.zeros(n)
    if n == 0:
        return smoothed
        
    # Kalman Filter initial setup
    x_est = data[0]
    p = initial_p
    q = 1e-5  # Process variance
    r = 0.1   # Measurement variance
    
    smoothed[0] = x_est
    for i in range(1, n):
        # Time Update
        p_prior = p + q
        # Measurement Update
        k_gain = p_prior / (p_prior + r)
        x_est = x_est + k_gain * (data[i] - x_est)
        p = (1 - k_gain) * p_prior
        smoothed[i] = x_est
        
    return smoothed

# ------------------------------------------------------------------
# 📊 MAIN LOGIC & STREAMLIT UI ENGINE
# ------------------------------------------------------------------

# Step 1: Input Array & Loop Length Verification
# (Ensuring data frames and state logs are perfectly aligned)

# Note: Make sure 'df_predict' is pre-loaded with 'a_Close', 'b_Kalman_Price', 
# 'Prob_Up', and 'Prob_Down' columns before calling this execution block.

if 'df_predict' in locals() or 'df_predict' in globals():
    
    # Configuration Bounds
    MAX_BUCKET = 5
    MIN_BUCKET = -5

    # Storage Log Initialization
    scores_log = []
    raw_weighted_momentum_log = []
    final_signals = []

    # Initial State Variables
    current_state = None  # Tracks trend state: "BUY", "SELL", or None
    accumulator = 0       # Points accumulator

    # Engine Computation Loop
    for idx, row in df_predict.iterrows():
        c_val = row['a_Close']
        k_price_val = row['b_Kalman_Price']
        p_up = row['Prob_Up']
        p_down = row['Prob_Down']
        
        # Core dynamic scoring mechanism (Accumulator Logic)
        if p_up > p_down:
            accumulator += 1
        elif p_down > p_up:
            accumulator -= 1
            
        # Upper and Lower Bound Saturation Checks
        accumulator = max(MIN_BUCKET, min(MAX_BUCKET, accumulator))
        scores_log.append(accumulator)

        # Raw Weighted Momentum (Close - Kalman)
        calc_raw_weighted = c_val - k_price_val
        raw_weighted_momentum_log.append(calc_raw_weighted)

        # Step 2 & 3: Saturation Limits State Machine Trigger
        if accumulator == MAX_BUCKET:
            current_state = "BUY"
            final_signals.append("🟢 STRONG BUY TREND (Max Locked [5/5])")
            
        elif accumulator == MIN_BUCKET:
            current_state = "SELL"
            final_signals.append("🔴 STRONG SELL TREND (Max Locked [-5/-5])")
            
        # Step 4 & 5: Intermediate State Degradation Matrix
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

    # Mapping secure array data back to pandas
    df_predict['d_ML_Signal'] = final_signals
    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 

    # Step 6: Second-Stage Double Kalman Filter Application
    # 🔥 AAPKI REGULAR REQUIREMENT: Weighted Momentum ke upar strictly Kalman 0.50 execution
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(
