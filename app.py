import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty BeES Custom Logic Bot", layout="wide")
st.title("🛡️ Nifty BeES: Your Custom Mathematical ML Engine")
st.write("🎯 **Logic:** a = Close, b = Kalman (0.001), c = Combined Matrix, d = ML on c")

# =====================================================================
# MATHEMATICAL ENGINE (b = Kalman Filter with 0.001 Process Noise)
# =====================================================================
def apply_kalman_filter_custom(price_array):
    x = price_array[0]
    p = 100.0  
    q = 0.001  # Fixed process noise requested by you
    r = 0.1    # Stable measurement noise
    filtered_prices = []
    for z in price_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_prices.append(x)
    return filtered_prices

def calculate_technical_context(df):
    # 1. VWAP
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-10)
    
    # 2. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    # 3. MACD
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    
    # 4. Plus/Minus DI
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    up_move = df['High'].diff()
    down_move = df['Low'].shift() - df['Low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    df['DMP_14'] = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / (df['ATR'] + 1e-10))
    df['DMN_14'] = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / (df['ATR'] + 1e-10))
    return df

# Fetch Live Data
with st.spinner("Processing your custom logic engine..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    df = yf.download("NIFTYBEES.NS", start="2025-01-01", end=end_date, interval="1h")
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    # a = Close Price
    df['a_Close'] = df['Close']
    
    # b = Kalman on a with 0.001 noise
    df['b_Kalman'] = apply_kalman_filter_custom(df['a_Close'].values)
    
    # Standard indicators for building context 'c'
    df = calculate_technical_context(df)
    
    # Target definition (Forward Trend Look-ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['VWAP', 'RSI_14', 'MACD', 'ATR', 'Target'], inplace=True)

# c = Combined Feature Matrix (AND context containing smooth trends and indicator filters)
features_c = ['Volume', 'VWAP', 'b_Kalman', 'MACD', 'RSI_14', 'ATR', 'DMP_14', 'DMN_14']
X = df[features_c]
y = df['Target']

# Train on 2025 data, Predict on 2026 data
train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

X_train, y_train = X[train_mask], y[train_mask]
X_test = X[test_mask]

if len(X_train) == 0 or len(X_test) == 0:
    st.error("Data tracking conflict. Please reboot app.")
else:
    # d = ML Engine running on optimized matrix 'c'
    model_d = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=12, random_state=42)
    model_d.fit(X_train, y_train)

    # Statistical Surety Probability
    probabilities = model_d.predict_proba(X_test)
    df_signals = df[test_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]
    df_signals['ML_Surety_%'] = np.maximum(df_signals['Prob_Up'], df_signals['Prob_Down']) * 100

    # High Surety Signal Filter (Set at a balanced 70% threshold for stable execution)
    df_signals['d_ML_Signal'] = "⚪ HOLD / NO ACTION"
    df_signals.loc[df_signals['Prob_Up'] >= 0.70, 'd_ML_Signal'] = "🟢 SURE BUY (Trend Up)"
    df_signals.loc[df_signals['Prob_Down'] >= 0.70, 'd_ML_Signal'] = "🔴 SURE SELL (Trend Down)"

    # Format Display Frame
    display_columns = ['a_Close', 'b_Kalman', 'VWAP', 'MACD', 'RSI_14', 'ML_Surety_%', 'd_ML_Signal']
    display_df = df_signals[display_columns].copy()

    for col in display_df.columns:
        if col != 'd_ML_Signal': 
            display_df[col] = display_df[col].round(2)
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

    # Render Clean Table View
    st.subheader(f"📋 Live Execution Table: 1 Jan 2026 to Today (Total: {len(display_df)} hours)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Verification Stats
    st.sidebar.header("⚙️ Engine Architecture")
    st.sidebar.info(f"""
    - **a (Close):** Raw Entry Price
    - **b (Kalman):** Smooth baseline at 0.001 noise
    - **c (Matrix):** Technical indicators synced
    - **d (ML Signal):** Random Forest on 'c'
    """)
    st.sidebar.write(f"🟢 Sure Buy Triggers: **{len(df_signals[df_signals['Prob_Up'] >= 0.70])}**")
    st.sidebar.write(f"🔴 Sure Sell Triggers: **{len(df_signals[df_signals['Prob_Down'] >= 0.70])}**")
