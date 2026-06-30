import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Custom Logic ML Bot", layout="wide")
st.title("🛡️ Nifty BeES: Pure a, b, c, d Matrix Engine")
st.write("🎯 **Formula:** a = Close Price | b = Kalman (0.001) | c = Combined Data | d = ML Signal")

# =====================================================================
# MATHEMATICAL ENGINE (b = Kalman Filter with 0.001 Process Noise)
# =====================================================================
def apply_kalman_filter_strict(price_array):
    x = price_array[0]
    p = 50.0  
    q = 0.001  # Fixed process noise requested by you
    r = 0.1    
    filtered_prices = []
    for z in price_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_prices.append(x)
    return filtered_prices

def calculate_technical_context(df):
    # Standard indicators computed behind the scenes to enrich 'c' matrix
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-10)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    return df

# Fetch Data
with st.spinner("Processing custom architecture matrices..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    df = yf.download("NIFTYBEES.NS", start="2025-01-01", end=end_date, interval="1h")
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    # a = Close Price
    df['a_Close'] = df['Close']
    
    # b = Kalman Filter on a (0.001)
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    
    # Background processing for structural support
    df = calculate_technical_context(df)
    
    # c_Combined value mapping for UI representation (Math relationship of raw and smooth price)
    df['c_Combined'] = (df['a_Close'] - df['b_Kalman']).round(4)
    
    # Target definition (Forward Look-ahead Trend)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['VWAP', 'RSI_14', 'MACD', 'Target'], inplace=True)

# Matrix setup for training (Inputs for ML Engine)
features_matrix = ['Volume', 'VWAP', 'b_Kalman', 'MACD', 'RSI_14', 'c_Combined']
X = df[features_matrix]
y = df['Target']

# Splitting Mask
train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

X_train, y_train = X[train_mask], y[train_mask]
X_test = X[test_mask]

if len(X_train) == 0 or len(X_test) == 0:
    st.error("Data Range mismatch. Please reboot application.")
else:
    # Model d execution on structured input c
    model_d = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=12, random_state=42)
    model_d.fit(X_train, y_train)

    probabilities = model_d.predict_proba(X_test)
    df_signals = df[test_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]
    
    # Setting threshold to 65% for logical yet steady confirmations
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    df_signals.loc[df_signals['Prob_Up'] >= 0.65, 'd_ML_Signal'] = "🟢 SURE BUY (Up Trend)"
    df_signals.loc[df_signals['Prob_Down'] >= 0.65, 'd_ML_Signal'] = "🔴 SURE SELL (Down Trend)"

    # Strict column filtering requested by you
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()

    # Formating decimal visualization
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

    # Main Screen Data View
    st.subheader(f"📋 Live Tracking Table: 1 Jan 2026 to Present (Total: {len(display_df)} hours)")
    
    # Render table containing only requested elements
    st.dataframe(display_df, use_container_width=True, height=750)

    # Download Option
    csv = display_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Structured Dataset (CSV)",
        data=csv,
        file_name='nifty_bees_abcd_structure.csv',
        mime='text/csv',
    )
