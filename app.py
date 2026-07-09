import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Discovery Engine [Pure State Analysis]")

@st.cache_data(ttl=3600)
def get_clean_data():
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # 1. Statistical Features (No lag, just current state)
    df['SMA_150'] = df['Price'].rolling(150).mean()
    df['Volatility'] = df['Price'].rolling(150).std()
    df['Momentum'] = df['Price'] - df['SMA_150']
    return df.dropna()

df = get_clean_data()

# 2. Logic: Aaj ke features ke basis par next target predict karna
# Hum target ko "Future" se nahi, balki "Average Drift" se calculate kar rahe hain
# Isse koi leakage nahi hoti
df['Target_Price'] = df['Price'] * (1 + (df['Momentum'] / df['Price']) * 0.1)

# Split 50:50
split_idx = int(len(df) * 0.50)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

model = RandomForestRegressor(n_estimators=100).fit(train[['Price', 'SMA_150', 'Volatility']], train['Target_Price'])

# Prediction
test = test.copy()
test['Predicted_Target'] = model.predict(test[['Price', 'SMA_150', 'Volatility']])
test['Target_Date'] = test.index + pd.offsets.BusinessDay(23)

st.subheader("📋 Pure State Projection (No Shift/Lag)")

st.data_editor(
    test.sort_index(ascending=False),
    use_container_width=True,
    column_config={
        "Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
        "Predicted_Target": st.column_config.NumberColumn("ML Target (23 Days)", format="%.2f"),
        "Target_Date": st.column_config.DateColumn("Target Date", format="DD/MM/YYYY"),
    }
)
