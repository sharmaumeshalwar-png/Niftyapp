import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Discovery Backtest Engine [50:50 Split]")

@st.cache_data(ttl=3600)
def get_backtest_data():
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # Features
    df['SMA_150'] = df['Price'].rolling(150).mean()
    df['Momentum'] = df['Price'] - df['SMA_150']
    
    # Target (150 hours later)
    df['Actual_Future_Price'] = df['Price'].shift(-150)
    return df.dropna()

df = get_backtest_data()

# 50:50 Split Logic
split_idx = int(len(df) * 0.50)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

# Training on the first 50%
features = ['Price', 'SMA_150', 'Momentum']
model = RandomForestRegressor(n_estimators=100).fit(train[features], train['Actual_Future_Price'])

# Predicting on the second 50% (Backtesting)
test = test.copy()
test['Predicted_Price'] = model.predict(test[features])
test['Projection_Date'] = test.index + pd.offsets.BusinessDay(23)

st.subheader("📋 Historical Backtest Table (Observation Mode)")
st.write("Yahan aap dekh sakte hain ki pichle 1 saal mein model ne har ghante kya predict kiya tha.")

st.data_editor(
    test.sort_index(ascending=False),
    use_container_width=True,
    height=600,
    column_config={
        "Price": st.column_config.NumberColumn("Price at Time", format="%.2f"),
        "Predicted_Price": st.column_config.NumberColumn("ML Prediction", format="%.2f"),
        "Actual_Future_Price": st.column_config.NumberColumn("Actual Result", format="%.2f"),
        "Projection_Date": st.column_config.DateColumn("Target Date", format="DD/MM/YYYY"),
    }
)
