import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🎯 ML Convergence: Bulletproof Audit")

@st.cache_data(ttl=3600)
def get_bulletproof_data():
    # 2-Year Data
    start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    df = yf.download("^NSEI", start=start, interval="1h", progress=False)
    
    # 1. FLAT COLUMN HACK: MultiIndex ko khatam karna
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. Convert to DataFrame if Series
    df = pd.DataFrame(df)
    
    # 3. Clean Data
    df = df.dropna(subset=['Close'])
    return df

df = get_bulletproof_data()

# ML Features
def create_features(df):
    data = df[['Close']].copy()
    for i in range(1, 9):
        data[f'lag_{i}'] = data['Close'].shift(i)
    return data.dropna()

ml_df = create_features(df)

# Split 50:50
split = len(ml_df) // 2
train = ml_df.iloc[:split]
test = ml_df.iloc[split:]

# ML Training
features = [f'lag_{i}' for i in range(1, 9)]
model = LinearRegression()
model.fit(train[features].values, train['Close'].values)

# Prediction
test['Predicted_Price'] = model.predict(test[features].values)

st.write(f"✅ Success: Loaded {len(ml_df)} hours of data.")
st.dataframe(test[['Close', 'Predicted_Price']].sort_index(ascending=False).head(20), use_container_width=True)

st.markdown("""
### 8-Step Verification (ML Convergence Logic):
1. **Flattening:** MultiIndex columns ko 'Close' aur 'lag' mein convert kiya (Error Fix).
2. **Feature Engineering:** 8-lag window ka matrix banaya.
3. **Data Integrity:** `dropna()` aur `values` conversion se `ValueError` khatam kiya.
4. **Training (50%):** Model ne pehle 1 saal ka historical trend seekha.
5. **Testing (50%):** Dusre saal ke data par model ko challenge kiya.
6. **Convergence:** `Predicted_Price` wo point hai jahan model ne price ko fix (Lock) kiya.
7. **Time Sequence:** Ab prediction aur original price ek hi timeline par hain.
8. **Final Output:** ML-driven fair value projection.
""")
