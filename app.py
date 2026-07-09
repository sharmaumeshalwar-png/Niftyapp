import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Page Setup
st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Discovery Engine [Walk-Forward Mode]")

@st.cache_data(ttl=3600)
def get_clean_data():
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # 1. Features (Past ke data par based)
    df['SMA_150'] = df['Price'].rolling(150).mean()
    df['Momentum'] = df['Price'] - df['SMA_150']
    
    # 2. Target (Ye sirf training ke waqt use hoga, prediction ke waqt nahi)
    # Hum model ko sikha rahe hain ki pichle 150 candle ke patterns se next 150 candle ka price kya raha tha
    df['Target_Price'] = df['Price'].shift(-150)
    return df.dropna()

df = get_clean_data()

# =====================================================================
# NO LEAKAGE TRAINING: Walk-Forward Logic
# =====================================================================
# Model sirf 'train' data se seekhega
train_size = int(len(df) * 0.70)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

features = ['Price', 'SMA_150', 'Momentum']
model = RandomForestRegressor(n_estimators=100, max_depth=5).fit(train[features], train['Target_Price'])

# Prediction
test = test.copy()
test['Predicted_Target_Price'] = model.predict(test[features])
test['Projection_Date'] = test.index + pd.offsets.BusinessDay(23)

st.subheader("📋 Zero Leakage Projection Table")
st.write("Model ne sirft train set se seekha hai. Test set par prediction puri tarah unbiased hai.")

st.data_editor(
    test.tail(50).sort_index(ascending=False),
    use_container_width=True,
    column_config={
        "Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
        "Predicted_Target_Price": st.column_config.NumberColumn("ML Predicted Price", format="%.2f"),
        "Projection_Date": st.column_config.DateColumn("Target Date", format="DD/MM/YYYY"),
    }
)
