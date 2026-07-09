import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50: Date-Stamped Geometry Audit")

@st.cache_data(ttl=3600)
def get_geometric_data():
    # 2 saal ka data
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = df[['Close']].ffill()
    df.columns = ['Price']
    
    emas = [20, 50, 100, 200]
    feature_cols = []
    
    for e in emas:
        ema_val = df['Price'].ewm(span=e).mean()
        df[f'Dist_{e}'] = df['Price'] / ema_val
        df[f'Slope_{e}'] = ema_val.pct_change(3)
        df[f'Vol_{e}'] = df['Price'].rolling(e).std() / ema_val
        feature_cols.extend([f'Dist_{e}', f'Slope_{e}', f'Vol_{e}'])
    
    # Target: Next hour prediction
    df['Target'] = df['Price'].shift(-1)
    return df.dropna(), feature_cols

data, features = get_geometric_data()

# 50:50 Split logic
mid_point = int(len(data) * 0.5)
train = data.iloc[:mid_point]
test = data.iloc[mid_point:]

# 12-Dimensional Tree Decoder
model = RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1)
model.fit(train[features], train['Target'])

# Prediction with explicit Date Tagging
test = test.copy()
test['Predicted_Price'] = model.predict(test[features])
test['Target_Date'] = test.index # Yahi aapki prediction date/time hai

st.subheader("📋 Audit: Which Date? Which Price?")
# Date-Target table
st.dataframe(test[['Target_Date', 'Price', 'Predicted_Price']].sort_index(ascending=False).head(20), use_container_width=True)
