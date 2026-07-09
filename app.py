import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Crash-Proof Engine")

@st.cache_data(ttl=60)
def get_live_data():
    # 3 mahine ka data lo taaki 150 candles ka window khali na ho
    raw = yf.download("^NSEI", period="3mo", interval="1h", progress=False)
    if raw.empty: return pd.DataFrame()
    
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # Dynamic Windowing: Agar data kam hai toh window size adjust karo
    window = min(50, len(df)//2)
    
    df['SMA'] = df['Price'].rolling(window=window).mean()
    df['Vol'] = df['Price'].rolling(window=window).std()
    
    # Target: Next candle price
    df['Target'] = df['Price'].shift(-1)
    return df.dropna()

data = get_live_data()

if data.empty or len(data) < 10:
    st.error("Data kam hai, calculation nahi ho pa rahi. Kuch der baad try karein.")
else:
    model = RandomForestRegressor(n_estimators=100)
    X = data[['SMA', 'Vol']]
    y = data['Target']
    
    model.fit(X, y)
    
    data['Prediction'] = model.predict(X)
    
    st.subheader("📋 Latest Market Data")
    st.dataframe(data.sort_index(ascending=False).head(10), use_container_width=True)
