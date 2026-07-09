import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Multi-Angle Geometry Decoder")

@st.cache_data(ttl=3600)
def get_geometric_data():
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = df[['Close']].ffill()
    df.columns = ['Price']
    
    emas = [20, 50, 100, 200]
    features = []
    
    for e in emas:
        ema_val = df['Price'].ewm(span=e).mean()
        # 1. Distance Angle (Price kitna door hai)
        df[f'Dist_{e}'] = df['Price'] / ema_val
        # 2. Slope Angle (EMA ka jhukav)
        df[f'Slope_{e}'] = ema_val.pct_change(3)
        # 3. Volatility Angle (EMA ke around kitna shor hai)
        df[f'Vol_{e}'] = df['Price'].rolling(e).std() / ema_val
        
        features.extend([f'Dist_{e}', f'Slope_{e}', f'Vol_{e}'])
    
    df['Target'] = df['Price'].shift(-1)
    return df.dropna(), features

data, feature_cols = get_geometric_data()

# 50:50 Split for 2-Year Audit
split = int(len(data) * 0.5)
train, test = data.iloc[:split], data.iloc[split:]

# 12-Dimensional Tree Decoder
model = RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1)
model.fit(train[feature_cols], train['Target'])

# Decoding
test = test.copy()
test['Decoded_Target'] = model.predict(test[feature_cols])

st.subheader("📋 12-Dimensional Geometry Audit")
st.dataframe(test.sort_index(ascending=False).head(20), use_container_width=True)
