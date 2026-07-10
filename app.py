import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("🎯 ML Convergence: Final Hardened Audit")

@st.cache_data(ttl=3600)
def get_hardened_data():
    # Load 2-year data
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    
    # Flatten Columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Data Cleaning: Har missing value ko handle karna
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

df = get_hardened_data()

# ML Features Setup
data = df[['Close']].copy()
for i in range(1, 9):
    data[f'lag_{i}'] = data['Close'].shift(i)

# NaN ko hatao jo shift() ke wajah se aate hain
data = data.dropna()

# 50:50 Split
split = len(data) // 2
train = data.iloc[:split]
test = data.iloc[split:]

# Ensure X and y are clean
X_train = train[[f'lag_{i}' for i in range(1, 9)]].values
y_train = train['Close'].values
X_test = test[[f'lag_{i}' for i in range(1, 9)]].values

# ML Training (Robust Mode)
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
test['Predicted_Price'] = model.predict(X_test)

st.write(f"✅ Data Sanitized: {len(test)} points audited.")
st.dataframe(test[['Close', 'Predicted_Price']].sort_index(ascending=False).head(20), use_container_width=True)
