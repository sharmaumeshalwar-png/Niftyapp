import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Discovery Engine [Cognitive Mode]")

@st.cache_data(ttl=3600)
def get_cognitive_data():
    raw = yf.download("^NSEI", period="1y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # 1. Cognitive Features (Momentum & Acceleration)
    df['SMA_150'] = df['Price'].rolling(150).mean()
    df['ROC'] = df['Price'].pct_change(20) # Rate of Change (Speed)
    df['Vol_Ratio'] = df['Price'].rolling(50).std() / df['Price'].rolling(200).std()
    
    # 2. "Mind" Logic: Price predict mat karo, "Deviation" predict karo
    # Hum model ko sikha rahe hain ki jab speed aur volatility aisi ho,
    # toh market apna mean (SMA) se kitna door jati hai.
    df['Deviation'] = (df['Price'] - df['SMA_150']) / df['SMA_150']
    
    df = df.dropna()
    return df

df = get_cognitive_data()

# Model "Deviation" predict kar raha hai, "Price" nahi.
X = df[['ROC', 'Vol_Ratio']]
y = df['Deviation']
model = RandomForestRegressor(n_estimators=100).fit(X, y)

# Prediction: Market kitni deviation show karega?
df['Predicted_Deviation'] = model.predict(X)
df['Discovery_Target'] = df['SMA_150'] * (1 + df['Predicted_Deviation'])
df['Projected_Date'] = df.index + pd.offsets.BusinessDay(23)

st.subheader("📋 Discovery: Deviation-Based Intelligence")
st.data_editor(
    df.sort_index(ascending=False).head(20),
    use_container_width=True,
    column_config={
        "Price": st.column_config.NumberColumn("Current", format="%.2f"),
        "Discovery_Target": st.column_config.NumberColumn("Discovery Target", format="%.2f"),
        "Projected_Date": st.column_config.DateColumn("Target Date", format="DD/MM/YYYY"),
    }
)
