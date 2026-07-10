import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty 2-Year Hybrid Engine", layout="wide")
st.title("📊 Nifty 50 Hybrid Engine: 2-Year 50:50 Split")

# =====================================================================
# MATHEMATICAL ENGINE: KALMAN FILTERS
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=100.0, q=0.0001, r=2.5): 
    if len(data_array) == 0: return []
    x = data_array[0]; p = initial_p
    filtered_values = []
    for z in data_array:
        p = p + q; k = p / (p + r); x = x + k * (z - x); p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

def apply_non_linear_kalman(data_array):
    if len(data_array) == 0: return []
    x = data_array[0]; p = 1.0; q = 0.05; r = 0.2
    filtered_values = []
    for z in data_array:
        p = p + q; k = p / (p + r); x = x + k * (z - x); p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("Loading 2-Year Nifty Data & Training Engine..."):
    # 2 Year Data Fetch
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    raw_df = yf.download("^NSEI", start=start_date, end=end_date, interval="1h")
    
    df = pd.DataFrame(index=raw_df.index)
    df['a_Close'] = raw_df['Close'].ffill()
    df['Open'] = raw_df['Open'].ffill(); df['High'] = raw_df['High'].ffill(); df['Low'] = raw_df['Low'].ffill()
    df.dropna(inplace=True)

    # Momentum Layers
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['Raw_Weighted_Momentum'] = df['a_Close'] - df['b_Kalman_Price']
    df['Weighted_Momentum'] = apply_kalman_filter_custom(df['Raw_Weighted_Momentum'].values, initial_p=0.50)
    df['Step_Momentum'] = np.round(apply_non_linear_kalman(df['Weighted_Momentum'].values))
    
    # ML Features
    df['Velocity'] = df['Weighted_Momentum'].diff(1).fillna(0)
    df['Acceleration'] = df['Velocity'].diff(1).fillna(0)
    df['Target'] = np.where(df['Weighted_Momentum'] > df['Weighted_Momentum'].shift(1), 1, 0)
    
    # Split Engine (50:50)
    df_clean = df.dropna().copy()
    split_idx = int(len(df_clean) * 0.50)
    train_df = df_clean.iloc[:split_idx]
    pred_df = df_clean.iloc[split_idx:]
    
    # Training
    features = ['Weighted_Momentum', 'Velocity', 'Acceleration']
    model = RandomForestClassifier(n_estimators=50, max_depth=3).fit(train_df[features], train_df['Target'])
    pred_df['ML_Prob'] = model.predict_proba(pred_df[features])[:, 1]
    
    # Display
    st.write(f"Total Data Points: {len(df_clean)} | Split Point: {split_idx}")
    display_cols = ['a_Close', 'Weighted_Momentum', 'Step_Momentum', 'ML_Prob']
    st.dataframe(pred_df[display_cols].sort_index(ascending=False), use_container_width=True)
