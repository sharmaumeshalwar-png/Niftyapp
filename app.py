import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(page_title="Nifty BeES ML Trading Bot", layout="wide")
st.title("📊 Nifty BeES ML Signal Engine (1-Hour Candle)")
st.write("Train Window: **1 Jan 2025 - 1 Jan 2026** | Signal Window: **1 Jan 2026 - Today**")

# =====================================================================
# PURE PYTHON MATH INDICATORS (No pandas-ta or filterpy needed!)
# =====================================================================
def apply_kalman_filter(price_array):
    x = price_array[0]
    p = 1000.0
    q = 0.01  
    r = 0.5   
    filtered_prices = []
    for z in price_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_prices.append(x)
    return filtered_prices

def calculate_indicators(df):
    # 1. VWAP
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # 2. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = exp1 - exp2
    
    # 4. ATR (14)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATRe_14'] = true_range.rolling(14).mean()
    
    # 5. Plus DI & Minus DI (Directional Movement)
    up_move = df['High'].diff()
    down_move = df['Low'].shift() - df['Low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    atr = df['ATRe_14']
    df['DMP_14'] = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / (atr + 1e-10))
    df['DMN_14'] = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / (atr + 1e-10))
    
    return df

# =====================================================================
# DATA FETCHING & PROCESSING
# =====================================================================
@st.cache_data(ttl=3600)
def load_and_process_data():
    df = yf.download("NIFTYBEES.NS", start="2025-01-01", interval="1h")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Process custom indicators
    df['Kalman_Price'] = apply_kalman_filter(df['Close'].values)
    df = calculate_indicators(df)
    
    df.dropna(inplace=True)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    return df

with st.spinner("Fetching Nifty BeES data and computing mathematical models..."):
    df = load_and_process_data()

# Define Features
features = ['Volume', 'VWAP', 'Kalman_Price', 'MACD_12_26_9', 'RSI_14', 'ATRe_14', 'DMP_14', 'DMN_14']
X = df[features]
y = df['Target']

# Splitting Data
train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# =====================================================================
# ML TRAINING & SELECTION
# =====================================================================
model_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42)

model_rf.fit(X_train, y_train)
model_gb.fit(X_train, y_train)

acc_rf = accuracy_score(y_train, model_rf.predict(X_train))
acc_gb = accuracy_score(y_train, model_gb.predict(X_train))

if acc_rf > acc_gb:
    best_model = model_rf
    winner_name = "Random Forest"
    winner_acc = acc_rf
else:
    best_model = model_gb
    winner_name = "Gradient Boosting"
    winner_acc = acc_gb

# Display Metrics on Sidebar
st.sidebar.header("🏆 ML Model Competition (2025 Data)")
st.sidebar.metric(label="Random Forest Accuracy", value=f"{acc_rf:.2%}")
st.sidebar.metric(label="Gradient Boosting Accuracy", value=f"{acc_gb:.2%}")
st.sidebar.success(f"Selected Best Model: **{winner_name}**")

# =====================================================================
# GENERATING OUT-OF-SAMPLE SIGNALS
# =====================================================================
df_signals = df[test_mask].copy()
df_signals['Predicted_Dir'] = best_model.predict(X_test)
df_signals['Signal'] = np.where(df_signals['Predicted_Dir'] == 1, "🟢 BUY (Long)", "🔴 SELL (Exit)")

# Clean display dataframe
display_df = df_signals[['Close', 'Kalman_Price', 'RSI_14', 'Signal']].copy()
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

# Layout Sections
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Recent Trading Signals (1 Jan 2026 - Present)")
    st.dataframe(display_df.tail(50), use_container_width=True)

with col2:
    st.subheader("ℹ️ Strategy Blueprint")
    st.info("""
    - **Interval:** 1-Hour Candlesticks
    - **Math Engine:** Custom Kalman Filter on Price
    - **Trend Features:** VWAP, MACD, ADX (DI+/DI-)
    - **Momentum & Volatility:** RSI, ATR, Volume
    """)
    
    csv = display_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Full Signal History (CSV)",
        data=csv,
        file_name='nifty_bees_ml_signals.csv',
        mime='text/csv',
    )

st.line_chart(df_signals[['Close', 'Kalman_Price']].tail(100))
st.caption("Showing last 100 hours: Real Close Price vs Kalman Smoothed Price")
