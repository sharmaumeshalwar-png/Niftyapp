import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(page_title="Nifty BeES ML Trading Bot", layout="wide")
st.title("📊 Nifty BeES ML Signal Engine (1-Hour Candle)")
st.write("Train Window: **1 Jan 2025 - 1 Jan 2026** | Signal Window: **1 Jan 2026 - Today**")

# =====================================================================
# CUSTOM KALMAN FILTER
# =====================================================================
def apply_kalman_filter(price_array):
    x = price_array[0]
    p = 1000.0
    q = 0.01  # Process noise
    r = 0.5   # Measurement noise
    filtered_prices = []
    for z in price_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_prices.append(x)
    return filtered_prices

# =====================================================================
# DATA FETCHING & PROCESSING
# =====================================================================
@st.cache_data(ttl=3600)  # Cache data for 1 hour to keep it fast
def load_and_process_data():
    # Fetch Data from Jan 2025 to Present
    df = yf.download("NIFTYBEES.NS", start="2025-01-01", interval="1h")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Apply Kalman Filter
    df['Kalman_Price'] = apply_kalman_filter(df['Close'].values)
    
    # Calculate Technical Indicators
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df.ta.macd(append=True)  # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    df.ta.rsi(append=True)   # RSI_14
    df.ta.atr(append=True)   # ATRe_14
    df.ta.adx(append=True)   # ADX_14, DMP_14 (+DI), DMN_14 (-DI)
    
    df.dropna(inplace=True)
    
    # Target: Next hour movement
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    return df

with st.spinner("Fetching data from Yahoo Finance and processing indicators..."):
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

# Choose Winner
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
# Format index for better visibility
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
    
    # Download Button
    csv = display_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Full Signal History (CSV)",
        data=csv,
        file_name='nifty_bees_ml_signals.csv',
        mime='text/csv',
    )

st.line_chart(df_signals[['Close', 'Kalman_Price']].tail(100))
st.caption("Showing last 100 hours: Real Close Price vs Kalman Smoothed Price")
