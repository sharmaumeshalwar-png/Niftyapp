import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# Streamlit Page Configuration
st.set_page_config(page_title="BTC HAM ML", layout="wide", initial_sidebar_state="collapsed")

st.title("🚀 BTC/USDT HAM & ML Predictor")

# Direct Binance REST API Fetcher with Safety Fallback
@st.cache_data(ttl=3600)
def fetch_btc_data_direct():
    symbol = "BTCUSDT"
    interval = "1h"
    
    # Try fetching historical data
    two_years_ago = datetime.utcnow() - timedelta(days=730)
    since_ms = int(two_years_ago.timestamp() * 1000)
    
    all_klines = []
    
    # Attempt batch fetching
    for _ in range(18): # Limit requests to avoid blocking
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={since_ms}&limit=1000"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json()
                if not data or not isinstance(data, list):
                    break
                all_klines.extend(data)
                since_ms = data[-1][6] + 1
                if len(data) < 1000:
                    break
            else:
                break
        except Exception:
            break

    # Fallback: If long historical fetch failed/blocked, fetch recent 1000 candles
    if len(all_klines) < 100:
        try:
            url_fallback = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1000"
            res_fb = requests.get(url_fallback, timeout=5)
            if res_fb.status_code == 200:
                all_klines = res_fb.json()
        except Exception:
            pass

    if not all_klines:
        return pd.DataFrame()

    # Parse JSON to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = df['open_time']
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df[['timestamp', 'close', 'volume']]

# Data Fetch Progress Spinner
with st.spinner("Binance API se Data Fetch Ho Raha Hai... Kripya Wait Karein..."):
    raw_df = fetch_btc_data_direct()

# Empty Data Guard Check (Prevents IndexError)
if raw_df.empty or len(raw_df) < 100:
    st.error("⚠️ Binance API se Data Fetch nahi ho paya. Kripya thodi der baad page refresh karein!")
    st.stop()

df = raw_df.copy()

# 1. Date & Time Column
df['Date_Time'] = pd.to_datetime(df['timestamp'], unit='ms')

# 2. Close Price Column
df['Close'] = df['close']

# 3. Kalman Filter Column (0.50 Gain directly on Close)
def apply_kalman(series, gain=0.50):
    estimates = []
    if len(series) == 0:
        return estimates
    curr = series.iloc[0]
    for p in series:
        curr = curr + gain * (p - curr)
        estimates.append(curr)
    return estimates

df['Kalman_0.50'] = apply_kalman(df['Close'], 0.50)

# Weighted Momentum Column (Close - Kalman)
df['Weighted_Momentum'] = df['Close'] - df['Kalman_0.50']

# 4. Host (Hurst Exponent Column - Rolling Window of 100)
def calculate_hurst(ts, max_lag=20):
    try:
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags if len(ts[lag:]) > 0]
        if len(tau) < 2 or any(t == 0 for t in tau):
            return 0.5
        poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
        return poly[0]
    except:
        return 0.5

df['Host'] = df['Close'].rolling(window=100).apply(lambda x: calculate_hurst(x.values), raw=False)

# 5. Volume Factor Column (Current Volume / 20-period Moving Avg Volume)
df['Avg_Volume_20'] = df['volume'].rolling(window=20).mean()
df['Volume_Factor'] = df['volume'] / df['Avg_Volume_20']

# 6. Host * Volume Factor Column
df['Host_x_Volume'] = df['Host'] * df['Volume_Factor']

# 7. Final HAM Column (Host_x_Volume * Weighted_Momentum)
df['Final_HAM'] = df['Host_x_Volume'] * df['Weighted_Momentum']

# Drop NaN values generated due to rolling calculations
df_clean = df.dropna().copy()

# Re-arranging EXACT Requested Column Sequence
final_cols = [
    'Date_Time', 
    'Close', 
    'Kalman_0.50', 
    'Weighted_Momentum', 
    'Host', 
    'Volume_Factor', 
    'Host_x_Volume', 
    'Final_HAM'
]
df_final = df_clean[final_cols].reset_index(drop=True)

# Machine Learning Implementation (50% Train / 50% Predict)
df_final['Target'] = np.where(df_final['Close'].shift(-1) > df_final['Close'], 1, 0)
features = ['Kalman_0.50', 'Weighted_Momentum', 'Host', 'Volume_Factor', 'Host_x_Volume', 'Final_HAM']

X = df_final[features][:-1]
y = df_final['Target'][:-1]

split_idx = int(len(X) * 0.50)

# 50-50 Split Data
X_train, X_predict = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_predict = y.iloc[:split_idx], y.iloc[split_idx:]

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 50% Predictions
predictions = model.predict(X_predict)

# Results Assembly
df_results = df_final.iloc[split_idx:-1].copy()
df_results['ML_Predicted_Direction'] = predictions
df_results['ML_Predicted_Direction'] = df_results['ML_Predicted_Direction'].map({1: '🟢 UP', 0: '🔴 DOWN'})

acc = accuracy_score(y_predict, predictions)

# Mobile Display Metrics
col1, col2 = st.columns(2)
col1.metric("Live Close", f"${df_final['Close'].iloc[-1]:,.2f}")
col2.metric("ML Accuracy (50%)", f"{acc * 100:.1f}%")

col3, col4 = st.columns(2)
col3.metric("Latest HAM", f"{df_final['Final_HAM'].iloc[-1]:.3f}")
col4.metric("Signal", df_results['ML_Predicted_Direction'].iloc[-1])

st.markdown("---")

# Plotly Interactive Chart
st.subheader("Price vs Kalman Filter (0.50)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_final['Date_Time'], y=df_final['Close'], name="Close", line=dict(color='gray', width=1)))
fig.add_trace(go.Scatter(x=df_final['Date_Time'], y=df_final['Kalman_0.50'], name="Kalman", line=dict(color='orange', width=2)))
fig.update_layout(height=350, template="plotly_dark", margin=dict(l=5, r=5, t=10, b=5))
st.plotly_chart(fig, use_container_width=True)

# Final Data Table
st.subheader("Results Table (50% Predict)")
st.dataframe(
    df_results[[
        'Date_Time', 
        'Close', 
        'Kalman_0.50', 
        'Weighted_Momentum', 
        'Host', 
        'Volume_Factor', 
        'Host_x_Volume', 
        'Final_HAM', 
        'ML_Predicted_Direction'
    ]].tail(100),
    use_container_width=True
)
