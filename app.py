import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50: 2-Year Full Geometry Backtest (50:50 Split)")

@st.cache_data(ttl=3600)
def get_full_data():
    # 2 saal ka full data download
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = df[['Close']].ffill()
    df.columns = ['Price']
    
    emas = [20, 50, 100, 200]
    features = []
    
    for e in emas:
        ema_val = df['Price'].ewm(span=e).mean()
        df[f'Dist_{e}'] = df['Price'] / ema_val
        df[f'Slope_{e}'] = ema_val.pct_change(3)
        df[f'Vol_{e}'] = df['Price'].rolling(e).std() / ema_val
        features.extend([f'Dist_{e}', f'Slope_{e}', f'Vol_{e}'])
    
    # Target: Next hour prediction
    df['Target'] = df['Price'].shift(-1)
    return df.dropna(), features

data, feature_cols = get_full_data()

# Strict 50:50 Split Logic
mid_point = int(len(data) * 0.5)
train = data.iloc[:mid_point]
test = data.iloc[mid_point:]

# Training on first 50% (Past 1 Year)
model = RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1)
model.fit(train[feature_cols], train['Target'])

# Prediction on second 50% (Future 1 Year Audit)
test = test.copy()
test['Predicted_Price'] = model.predict(test[feature_cols])
test['Date'] = test.index

st.subheader("📋 2-Year Performance Audit (Split: 50% Train / 50% Test)")
st.write(f"Training Range: {train.index[0].date()} to {train.index[-1].date()}")
st.write(f"Testing Range (Audit): {test.index[0].date()} to {test.index[-1].date()}")

st.dataframe(test[['Date', 'Price', 'Predicted_Price']].sort_index(ascending=False), use_container_width=True)
