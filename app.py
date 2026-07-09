import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Discovery Engine [Live-Locked Mode]")

@st.cache_data(ttl=3600)
def get_live_data():
    # .NS suffix add kiya hai for reliable NSE data
    raw = yf.download("^NSEI.NS", period="1y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # Features
    df['SMA_150'] = df['Price'].rolling(150).mean()
    
    # Target: Sirf wo rows jahan hamare paas 150 ghante baad ka data REAL mein hai
    # 'shift(-150)' lagane par jahan data nahi hai, wo dropna() se hat jayega
    df['Actual_Future'] = df['Price'].shift(-150)
    return df.dropna()

df = get_live_data()

# Model sirf 'Completed History' par train hoga
model = RandomForestRegressor(n_estimators=100).fit(df[['Price', 'SMA_150']], df['Actual_Future'])

# Prediction
df['Prediction'] = model.predict(df[['Price', 'SMA_150']])
df['Target_Date'] = df.index + pd.offsets.BusinessDay(23)

st.subheader("📋 Final Verified Report (Live Data Only)")

# Last 20 rows jahan tak actual data actual mein available hai
st.data_editor(
    df.sort_index(ascending=False).head(20),
    use_container_width=True,
    column_config={
        "Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
        "Prediction": st.column_config.NumberColumn("Model Target", format="%.2f"),
        "Actual_Future": st.column_config.NumberColumn("Actual Market Result", format="%.2f"),
        "Target_Date": st.column_config.DateColumn("Projection Date", format="DD/MM/YYYY"),
    }
)
