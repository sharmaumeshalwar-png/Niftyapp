import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty BeES Pro Bot", layout="wide")
st.title("🛡️ Nifty BeES Smart ML Engine (With Confidence Control)")

# Sidebar Configuration for Surety Control
st.sidebar.header("🎯 Target Control Panel")
confidence_level = st.sidebar.slider(
    "Minimum ML Surety (%)", 
    min_value=55, 
    max_value=85, 
    value=65,  # Default set to 65% for optimal signals
    step=5,
    help="Surety jitni kam hogi, signals utne zyada aayenge. 65%-70% sabse best aur accurate hota hai."
)

st.write(f"📊 Showing signals where Machine Learning Model is at least **{confidence_level}%** Sure.")

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
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    df = yf.download("NIFTYBEES.NS", start="2025-01-01", end=end_date, interval="1h")
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    df['Kalman_Price'] = apply_kalman_filter(df['Close'].values)
    df = calculate_indicators(df)
    
    df['Target'] = np.where(df['Close'].shift(-3) > df['Close'], 1, 0)
    df.dropna(subset=['VWAP', 'RSI_14', 'MACD_12_26_9', 'ATRe_14', 'Target'], inplace=True)

# Features Matrix Setup
features = ['Volume', 'VWAP', 'Kalman_Price', 'MACD_12_26_9', 'RSI_14', 'ATRe_14', 'DMP_14', 'DMN_14']
X = df[features]
y = df['Target']

train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

X_train, y_train = X[train_mask], y[train_mask]
X_test = X[test_mask]

if len(X_train) == 0 or len(X_test) == 0:
    st.error("Data range issue. Re-triggering...")
else:
    # Model
    model = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)

    # Probabilities
    probabilities = model.predict_proba(X_test)
    df_signals = df[test_mask].copy()
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Convert Slider value to decimal decimal threshold
    threshold = confidence_level / 100.0

    # Dynamic Filtering Rules
    df_signals['Signal'] = "⚪ WAIT (Low Confidence)"
    df_signals['ML_Surety_%'] = np.maximum(df_signals['Prob_Up'], df_signals['Prob_Down']) * 100

    df_signals.loc[df_signals['Prob_Up'] >= threshold, 'Signal'] = "🟢 STRONG BUY (Full Sure)"
    df_signals.loc[df_signals['Prob_Down'] >= threshold, 'Signal'] = "🔴 STRONG SELL (Full Sure)"

    # Formating Outputs
    display_columns = ['Close', 'Kalman_Price', 'VWAP', 'MACD_12_26_9', 'RSI_14', 'ML_Surety_%', 'Signal']
    display_df = df_signals[display_columns].copy()

    for col in display_df.columns:
        if col != 'Signal': 
            display_df[col] = display_df[col].round(2)
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

    # Render Data
    st.dataframe(display_df, use_container_width=True, height=700)

    # Sidebar Filter Stats
    active_buys = len(df_signals[df_signals['Prob_Up'] >= threshold])
    active_sells = len(df_signals[df_signals['Prob_Down'] >= threshold])
    
    st.sidebar.subheader("📊 Live Signal Counter")
    st.sidebar.write(f"🟢 Active Buys: **{active_buys}**")
    st.sidebar.write(f"🔴 Active Sells: **{active_sells}**")
