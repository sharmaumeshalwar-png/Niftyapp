import streamlit as st
import pandas as pd
import numpy as np

# --- Dummy Kalman Filter Function (For demonstration, replace with your actual function) ---
def apply_kalman_filter_custom(values, initial_p=0.50):
    """
    Strict Kalman filter execution with initial state covariance p = 0.50
    """
    n = len(values)
    filtered_values = np.zeros(n)
    
    # Kalman initialization variables
    x_est = values[0] if n > 0 else 0.0
    p = initial_p
    q = 1e-5  # Process variance
    r = 0.01  # Measurement variance
    
    for k in range(n):
        if np.isnan(values[k]):
            filtered_values[k] = x_est
            continue
        # Prediction update
        p_prior = p + q
        # Measurement update
        k_gain = p_prior / (p_prior + r)
        x_est = x_est + k_gain * (values[k] - x_est)
        p = (1 - k_gain) * p_prior
        filtered_values[k] = x_est
        
    return filtered_values

# --- Main Engine Processing Block ---
def process_nifty_engine_data(df_predict, scores_log, raw_weighted_momentum_log):
    """
    Processes the dynamic Nifty index pipeline following an 8-Step Verification structure.
    """
    # Step 1: Input Assignment
    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 

    # Step 2: Signal Smoothing Execution (Strict Kalman 0.50 Setup)
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(
        df_predict['Raw_Weighted_Momentum'].values, 
        initial_p=0.50
    )

    # Step 3: Column Filtering (Isolating core features)
    clean_display_cols = ['a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Weighted_Momentum', 'd_ML_Signal']
    display_df = df_predict[clean_display_cols].copy()
    
    # Step 4: Data Type Casts & Precisions (With NaN safety)
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    
    # Safety Hack: .astype(int) handles NaN poorly, filling with 0 or forward fill first
    display_df['Accumulator_Score'] = display_df['Accumulator_Score'].fillna(0).astype(int)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    
    # Step 5: Temporal Ordering (Latest rows on top)
    display_df = display_df.sort_index(ascending=False)
    
    # Step 6: Index Datetime Serialization
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    # Step 7 & 8: UI Component Binding & Native Dynamic Render
    st.subheader(f"📋 Live 1-Hour Nifty Index Engine (Double Kalman 0.50 Setup)")
    st.dataframe(display_df, use_container_width=True, height=750)
    
    return display_df

# --- Streamlit Execution Mock (For verification testing) ---
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    
    # Creating Mock Data to simulate live index behavior
    date_range = pd.date_range(start="2026-07-01", periods=100, freq="1h")
    mock_df = pd.DataFrame({
        'a_Close': np.random.uniform(24000, 24500, size=100),
        'b_Kalman_Price': np.random.uniform(24000, 24500, size=100),
        'Prob_Up': np.random.uniform(0, 1, size=100),
        'Prob_Down': np.random.uniform(0, 1, size=100),
        'd_ML_Signal': np.random.choice(['BUY', 'SELL', 'HOLD'], size=100)
    }, index=date_range)
    
    mock_scores = np.random.randint(-5, 6, size=100)
    mock_momentum = np.random.uniform(-2, 2, size=100)
    
    # Execute the Engine Pipeline
    process_nifty_engine_data(mock_df, mock_scores, mock_momentum)
