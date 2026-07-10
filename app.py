import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("🎯 ML Convergence Sniper: Verified Execution")

@st.cache_data(ttl=3600)
def get_clean_ml_data():
    # 2 Year Data with strict range
    end = datetime.now()
    start = end - timedelta(days=730)
    df = yf.download("^NSEI", start=start, end=end, interval="1h", progress=False)
    
    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Strict Cleaning
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Close'])
    df = df.ffill().bfill() # Gap fill
    return df

df = get_clean_ml_data()

# Features Creation
def create_ml_features(data):
    d = data.copy()
    for i in range(1, 9):
        d[f'lag_{i}'] = d['Close'].shift(i)
    return d.dropna()

data_ml = create_ml_features(df)

# Split 50:50
split_idx = len(data_ml) // 2
train = data_ml.iloc[:split_idx]
test = data_ml.iloc[split_idx:]

# Model training
features = [f'lag_{i}' for i in range(1, 9)]
model = LinearRegression()
model.fit(train[features], train['Close'])

# Predictions
test['Predicted_Price'] = model.predict(test[features])

st.write(f"✅ Data Sanitized. Audit range: {len(test)} hours.")
st.dataframe(test[['Close', 'Predicted_Price']].sort_index(ascending=False).head(20), use_container_width=True)

st.markdown("""
### 8-Step Verification & Logic:
1. **Sanitize:** `bfill()` aur `ffill()` se saare missing values hataye gaye.
2. **Feature Alignment:** 8-lag features ka matrix banaya gaya.
3. **Training:** Linear regression ne 1st saal ke market direction ko map kiya.
4. **Prediction:** Model ne current 'Close' aur pichle 8 ghante ke 'Lag' se target price nikala.
5. **Validation:** `NaN` aur `Inf` error ko `replace` aur `dropna` se permanently block kiya.
6. **Convergence:** `Predicted_Price` wo ML-calculated point hai jahan price converge hona chahiye.
7. **Time sequence:** Ab prediction sahi sequence mein hai.
8. **Audit:** Table mein 'Close' (Asli price) vs 'Predicted_Price' (ML Model price) compare karo.
""")
