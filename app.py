import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Page Setup
st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Discovery Engine [Hard Leakage Protection]")

@st.cache_data(ttl=3600)
def get_clean_data():
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # 1. Past Features
    df['SMA_150'] = df['Price'].rolling(150).mean()
    df['Momentum'] = df['Price'] - df['SMA_150']
    
    # 2. TARGET: Humein 150 ghante baad ka price chahiye.
    # Lekin hum train karte waqt "Future" ko training data se exclude karenge.
    df['Future_Price'] = df['Price'].shift(-150)
    return df.dropna()

df = get_clean_data()

# =====================================================================
# HARD LEAKAGE PROTECTION: Training on PRE-EXISTING data only
# =====================================================================
# Hum test set ko "Cut-off" kar rahe hain aaj ki date par
# Taaki model kabhi bhi aage ki date na dekh sake
cutoff_date = df.index[-150] 
train = df[df.index < cutoff_date]
test = df[df.index >= cutoff_date]

features = ['Price', 'SMA_150', 'Momentum']
model = RandomForestRegressor(n_estimators=100, max_depth=5).fit(train[features], train['Future_Price'])

# Prediction
test = test.copy()
test['Predicted_Target_Price'] = model.predict(test[features])
test['Projection_Date'] = test.index + pd.offsets.BusinessDay(23)

st.subheader("📋 Unbiased Projection (Training restricted to past only)")

st.data_editor(
    test.sort_index(ascending=False),
    use_container_width=True,
    column_config={
        "Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
        "Predicted_Target_Price": st.column_config.NumberColumn("ML Predicted Price", format="%.2f"),
        "Projection_Date": st.column_config.DateColumn("Target Date", format="DD/MM/YYYY"),
    }
)
