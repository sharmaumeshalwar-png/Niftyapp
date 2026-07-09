import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# Page Setup
st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 ML Discovery Engine [Live Data Enabled]")

# =====================================================================
# MATH & DATE ENGINE (Updated for Current Date)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=0.50):
    x, p = data_array[0], initial_p
    q, r = 0.0001, 2.5
    res = []
    for z in data_array:
        p += q
        k = p / (p + r)
        x += k * (z - x)
        p = (1 - k) * p
        res.append(x)
    return np.array(res)

@st.cache_data(ttl=3600)
def get_ml_data():
    # Dynamic Date Range Logic
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730) # 2 Years
    
    raw = yf.download("^NSEI", start=start_date, end=end_date, interval="1h")
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill()
    
    # Mathematical Features
    df['Kalman'] = apply_kalman_filter_custom(df['Price'].values)
    df['Weighted_Momentum'] = apply_kalman_filter_custom((df['Price'] - df['Kalman']).values)
    df['Step_Momentum'] = np.round(apply_kalman_filter_custom(df['Weighted_Momentum'].values) * 10)
    df['Past_150_Diff'] = df['Price'] - df['Price'].shift(150)
    
    # 150 Candles = ~23 Trading Days
    df['Target'] = df['Price'].shift(-150)
    df['Predicted_Date'] = df.index + pd.offsets.BusinessDay(23)
    return df.dropna()

df = get_ml_data()

# ML Training (50:50 Split)
split_idx = int(len(df) * 0.50)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]
features = ['Price', 'Kalman', 'Weighted_Momentum', 'Step_Momentum', 'Past_150_Diff']
model = RandomForestRegressor(n_estimators=100, max_depth=5).fit(train[features], train['Target'])
test['Predicted_150_Candle_Price'] = model.predict(test[features])

# =====================================================================
# DASHBOARD
# =====================================================================
st.subheader("📋 Discovery Engine (Updated to July 2026)")

st.data_editor(
    test.sort_index(ascending=False),
    use_container_width=True,
    height=600,
    column_config={
        "Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
        "Predicted_150_Candle_Price": st.column_config.NumberColumn("ML Pred. Target", format="%.2f"),
        "Predicted_Date": st.column_config.DateColumn("Projection Date", format="DD/MM/YYYY"),
    }
)
