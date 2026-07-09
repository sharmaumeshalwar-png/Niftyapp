import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Cognitive Backtest [50:50 Split]")

@st.cache_data(ttl=3600)
def get_backtest_data():
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # Cognitive Features
    df['SMA_150'] = df['Price'].rolling(150).mean()
    df['ROC'] = df['Price'].pct_change(20)
    df['Vol_Ratio'] = df['Price'].rolling(50).std() / df['Price'].rolling(200).std()
    
    # Target: Deviation from Mean
    df['Actual_Deviation'] = (df['Price'] - df['SMA_150']) / df['SMA_150']
    
    return df.dropna()

df = get_backtest_data()

# 50:50 Split Logic
split_idx = int(len(df) * 0.50)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

# Model Training
X_train = train[['ROC', 'Vol_Ratio']]
y_train = train['Actual_Deviation']
model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)

# Backtesting
test = test.copy()
test['Predicted_Deviation'] = model.predict(test[['ROC', 'Vol_Ratio']])
test['Discovery_Target'] = test['SMA_150'] * (1 + test['Predicted_Deviation'])
test['Projected_Date'] = test.index + pd.offsets.BusinessDay(23)

st.subheader("📋 50:50 Split Cognitive Audit")

st.data_editor(
    test.sort_index(ascending=False),
    use_container_width=True,
    column_config={
        "Price": st.column_config.NumberColumn("Actual Price", format="%.2f"),
        "Discovery_Target": st.column_config.NumberColumn("Model Discovery Target", format="%.2f"),
        "Projected_Date": st.column_config.DateColumn("Target Date", format="DD/MM/YYYY"),
    }
)
