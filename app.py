import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Nifty Kalman Engine", layout="wide")
st.title("📊 Nifty 50 Live 1-Hour Hybrid Double Kalman [0.50 Engine]")
st.write("🎯 **Aapki Custom Setting:** Strictly Only Nifty 50 Index Data + Strictly Past 25-Candle Window + Only 0.50 Kalman Layer")

# =====================================================================
# MATHEMATICAL ENGINE 1: LINEAR KALMAN FILTER (Price Layer)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=100.0): 
    if len(data_array) == 0: 
        return []
    x = data_array[0]
    p = initial_p  
    q = 0.0001     
    r = 2.5        
    
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

# =====================================================================
# MATHEMATICAL ENGINE 2: NON-LINEAR KALMAN FILTER (0.50 Baseline Layer)
# =====================================================================
def apply_non_linear_kalman_momentum(data_array):
    if len(data_array) == 0: 
        return []
    x = data_array[0]
    p = 0.50   
    q = 0.05   
    r = 0.2    
    
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("Processing 25-Candle Kalman Chains..."):
    # Download data safely
    raw_df = yf.download("^NSEI", period="1mo", interval="1h")
    
    if raw_df.empty:
        st.error("YFinance API Timeout. Please refresh.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in raw_df.columns:
            df[col] = raw_df[col].ffill()

    df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)

    # Strictly past 25 candles window
    if len(df) >= 25:
        df = df.tail(25).copy()

    # Core Kalman Calculation
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=100.0)
    
    # Weighted Momentum (Price - Kalman Price)
    df['Raw_Weighted_Momentum'] = df['a_Close'] - df['b_Kalman_Price']
    df['Weighted_Momentum'] = apply_kalman_filter_custom(df['Raw_Weighted_Momentum'].values, initial_p=0.50)
    
    # Non-Linear Step Momentum
    df['Step_Momentum'] = np.round(apply_non_linear_kalman_momentum(df['Weighted_Momentum'].values))

    # Engine Safe Baseline Fields (No ML Model = 0% Crash Chance)
    df['Prob_Up'] = 0.50
    df['Prob_Down'] = 0.50
    df['ML_WM_Linear_Prob'] = 0.50
    df['ML_WM_NonLinear_Prob'] = 0.50
    df['Accumulator_Score'] = 0
    df['d_ML_Signal'] = "🔄 ENGINE ACTIVE (0.50 Pure Kalman Locked)"
    df['Trap_Status'] = "TREND VALID"

    # Display Columns Mapping
    clean_cols = [
        'a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 
        'Accumulator_Score', 'Weighted_Momentum', 'Step_Momentum', 
        'ML_WM_Linear_Prob', 'ML_WM_NonLinear_Prob', 'd_ML_Signal', 'Trap_Status'
    ]
    
    display_df = df[clean_cols].copy().sort_index(ascending=False)
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2)
    display_df['Step_Momentum'] = display_df['Step_Momentum'].astype(int)
    
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader("📋 Nifty 50 Strict 25-Candle Pure Kalman Dashboard")
    st.dataframe(display_df, use_container_width=True)
