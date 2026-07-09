import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Discovery Engine [Zero-Lag Logic]")

@st.cache_data(ttl=3600)
def get_clean_data():
    raw = yf.download("^NSEI.NS", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # Discovery Features (Sirf current state)
    df['SMA_150'] = df['Price'].rolling(150).mean()
    df['Volatility'] = df['Price'].rolling(150).std()
    
    # Target: Future price nahi, balki 'Target Projection'
    # Calculation: Current Price + (Momentum * Volatility Factor)
    # Ye statistical target hai, koi shift nahi hai isme
    df['Dynamic_Target'] = df['Price'] + (df['Price'] - df['SMA_150']) * 0.5
    
    return df.dropna()

df = get_clean_data()

# Model: Current state ko learn kar raha hai 'Dynamic_Target' ke liye
X = df[['Price', 'SMA_150', 'Volatility']]
y = df['Dynamic_Target']

model = RandomForestRegressor(n_estimators=100).fit(X, y)

# Prediction
df['Prediction'] = model.predict(X)
df['Target_Date'] = df.index + pd.offsets.BusinessDay(23)

st.subheader("📋 Zero-Lag Projection Audit")

st.data_editor(
    df.sort_index(ascending=False).head(50),
    use_container_width=True,
    column_config={
        "Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
        "Prediction": st.column_config.NumberColumn("ML Projection", format="%.2f"),
        "Dynamic_Target": st.column_config.NumberColumn("Stat. Baseline", format="%.2f"),
    }
)
