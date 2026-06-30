import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Nifty BeES High-Confidence Bot", layout="wide")
st.title("🛡️ Nifty BeES Ultra-High Confidence ML Engine")
st.write("🎯 **Strategy:** Low Frequency, High Accuracy (Signals trigger only when ML is >= 80% Sure)")

# Pure Python Indicators Math
def apply_kalman_filter(price_array):
    x = price_array[0]
    p = 50.0  
    q = 0.001 
    r = 0.1   
    filtered_prices = []
    for z in price_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_prices.append(x)
    return filtered_prices

def calculate_indicators(df):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-10)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    df['MACD_12_26_9'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATRe_14'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    up_move = df['High'].diff()
    down_move = df['Low'].shift() - df['Low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    df['DMP_14'] = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / (df['ATRe_14'] + 1e-10))
    df['DMN_14'] = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / (df['ATRe_14'] + 1e-10))
    return df

# Fetch Data
with st.spinner("Analyzing market patterns..."):
    today_date = datetime.now().strftime('%Y-%m-%d')
    df = yf.download("NIFTYBEES.NS", start="2025-01-01", end=today_date, interval="1h")
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    df['Kalman_Price'] = apply_kalman_filter(df['Close'].values)
    df = calculate_indicators(df)
    df.dropna(inplace=True)

# Trend Target
df['Target'] = np.where(df['Close'].shift(-3) > df['Close'], 1, 0)

features = ['Volume', 'VWAP', 'Kalman_Price', 'MACD_12_26_9', 'RSI_14', 'ATRe_14', 'DMP_14', 'DMN_14']
X = df[features]
y = df['Target']

train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

# Balanced Model Configuration
model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=10, random_state=42)
model.fit(X[train_mask], y[train_mask])

# Deploy and Get Probabilities (Surety Numbers)
df_signals = df[test_mask].copy()
probabilities = model.predict_proba(X[test_mask])

# Extract probability for Down (Class 0) and Up (Class 1)
df_signals['Prob_Down'] = probabilities[:, 0]
df_signals['Prob_Up'] = probabilities[:, 1]

# Default state is Wait
df_signals['Signal'] = "⚪ WAIT (Low Confidence)"
df_signals['ML_Surety_%'] = np.maximum(df_signals['Prob_Up'], df_signals['Prob_Down']) * 100

# STRICT RULES: Trigger only when confidence is >= 80%
df_signals.loc[df_signals['Prob_Up'] >= 0.80, 'Signal'] = "🟢 STRONG BUY (Full Sure)"
df_signals.loc[df_signals['Prob_Down'] >= 0.80, 'Signal'] = "🔴 STRONG SELL (Full Sure)"

# Format Display DataFrame
display_columns = ['Close', 'Kalman_Price', 'VWAP', 'MACD_12_26_9', 'RSI_14', 'ML_Surety_%', 'Signal']
display_df = df_signals[display_columns].copy()

for col in display_df.columns:
    if col != 'Signal': 
        display_df[col] = display_df[col].round(2)
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

# Main Screen Output
st.subheader(f"📋 1 Jan 2026 to Present - Filtered Signals")
st.dataframe(display_df, use_container_width=True, height=700)

# Quick Stats on Sidebar
total_signals = len(df_signals)
strong_buys = len(df_signals[df_signals['Prob_Up'] >= 0.80])
strong_sells = len(df_signals[df_signals['Prob_Down'] >= 0.80])

st.sidebar.header("📊 Signal Filtration Statistics")
st.sidebar.write(f"Total Hours
