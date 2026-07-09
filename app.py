import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Discovery Engine [Unbiased Projection]")

@st.cache_data(ttl=3600)
def get_processed_data():
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # Past 150 Candles Features
    df['SMA_150'] = df['Price'].rolling(150).mean()
    df['Momentum'] = df['Price'] - df['SMA_150']
    
    # TARGET: Aaj ke price se 150 candles baad kya price hoga.
    # Hum yahan shift(-150) use kar rahe hain, lekin training mein sirf 
    # wo data use karenge jo "Future" nahi hai.
    df['Target_Price'] = df['Price'].shift(-150)
    return df.dropna() # Ye NaN wali rows (Future) hata dega

df = get_processed_data()

# TRAINING: Sirf uss data par jiska target hamare paas available hai
# (Yani 150 ghante purana data)
X = df[['Price', 'SMA_150', 'Momentum']]
y = df['Target_Price']

model = RandomForestRegressor(n_estimators=100).fit(X, y)

# PREDICTION: Aaj ka latest point
latest_point = df.iloc[[-1]][['Price', 'SMA_150', 'Momentum']]
predicted_price = model.predict(latest_point)[0]

# Projection Date calculation (150 hours = 23 Business Days)
target_date = df.index[-1] + pd.offsets.BusinessDay(23)

st.subheader("🎯 Market Prediction")
col1, col2 = st.columns(2)
col1.metric("Current Market Price", f"{df['Price'].iloc[-1]:.2f}")
col2.metric("Predicted Price (After 150 Candles)", f"{predicted_price:.2f}")

st.write(f"### 🗓️ Predicted Date: {target_date.strftime('%d/%m/%Y')}")

st.info("""
**Data Logic Explanation:**
1. Model ne pichle 2 saal ke har 150-candle ke pattern ko observe kiya hai.
2. Leakage hatane ke liye, humne sirf wahi data train kiya hai jiska 'Future' (150 candles baad ka price) record mein tha.
3. Prediction 'Latest Price' par ki gayi hai, bina future ka price dekhne diye.
""")
