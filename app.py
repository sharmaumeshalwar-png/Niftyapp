import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
# XGBoost light and fast hai, memory ka load nahi leta
from xgboost import XGBRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Pro-Engine [XGBoost Optimized]")

@st.cache_data(ttl=3600)
def get_pro_data():
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # Feature Engineering (Engineered for 'Smart' Discovery)
    df['SMA_Trend'] = df['Price'].rolling(50).mean() / df['Price'].rolling(200).mean()
    df['Volatility'] = df['Price'].rolling(30).std()
    df['Momentum'] = df['Price'].pct_change(10)
    df['Mean_Reversion'] = df['Price'] - df['Price'].rolling(100).mean()
    
    # Target
    df['Target'] = df['Price'].pct_change(150).shift(-150)
    return df.dropna()

data = get_pro_data()

# Model: 500 'Boosting Rounds' (Ye 500 trees ki tarah kaam karta hai, har tree pichli galti sudharti hai)
# Yeh memory mein halka hai par logic mein bahut gehra
model = XGBRegressor(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=6,
    n_jobs=-1
)

X = data[['SMA_Trend', 'Volatility', 'Momentum', 'Mean_Reversion']]
y = data['Target']

model.fit(X, y)

# Prediction
data['Predicted_Move'] = model.predict(X)
data['Smart_Target'] = data['Price'] * (1 + data['Predicted_Move'])

st.subheader("📋 Pro-Grade Discovery Audit")
st.dataframe(data.sort_index(ascending=False).head(50), use_container_width=True)
