import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("🎯 ML Convergence Sniper: 2-Year Full Audit")

@st.cache_data(ttl=3600)
def get_ml_data():
    # 2 Year Batching (Server Timeout avoid karne ke liye)
    end = datetime.now()
    start = end - timedelta(days=730)
    # Batch request for stability
    df = yf.download("^NSEI", start=start, end=end, interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.ffill().dropna()

df = get_ml_data()

# 50:50 Split for ML Training
split = len(df) // 2
train_df = df.iloc[:split]
test_df = df.iloc[split:].copy()

# ML Features Engineering
# Hum 'Close' price predict karenge pichle 8 ghante ke features se
def create_features(data):
    d = data.copy()
    for i in range(1, 9):
        d[f'lag_{i}'] = d['Close'].shift(i)
    return d.dropna()

train_feat = create_features(train_df)
test_feat = create_features(test_df)

X_train = train_feat[[f'lag_{i}' for i in range(1, 9)]]
y_train = train_feat['Close']
X_test = test_feat[[f'lag_{i}' for i in range(1, 9)]]

# ML Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
test_feat['Predicted_Price'] = model.predict(X_test)
test_feat['Target_Time'] = test_feat.index

st.write(f"📊 ML Model Trained on: {len(train_feat)} hours | Audit: {len(test_feat)} hours")
st.dataframe(test_feat[['Close', 'Predicted_Price', 'Target_Time']].sort_index(ascending=False).head(20), use_container_width=True)
