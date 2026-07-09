import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# Page Setup
st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Discovery Engine [Fixed]")

# =====================================================================
# MATH & DATA ENGINE (Error-Proof)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=0.50):
    if len(data_array) == 0: return np.array([])
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
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    raw = yf.download("^NSEI", start=start_date, end=end_date, interval="1h")
    
    # 1. Error Handling: Check if data is empty
    if raw.empty:
        return pd.DataFrame()
        
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill()
    df = df.dropna() # Remove NaNs
    
    # 2. Safety Check: Data length must be sufficient for 150-candle logic
    if len(df) < 200:
        return pd.DataFrame()

    df['Kalman'] = apply_kalman_filter_custom(df['Price'].values)
    df['Weighted_Momentum'] = apply_kalman_filter_custom((df['Price'] - df['Kalman']).values)
    df['Step_Momentum'] = np.round(apply_kalman_filter_custom(df['Weighted_Momentum'].values) * 10)
    df['Past_150_Diff'] = df['Price'] - df['Price'].shift(150)
    
    df['Target'] = df['Price'].shift(-150)
    df['Predicted_Date'] = df.index + pd.offsets.BusinessDay(23)
    return df.dropna()

df = get_ml_data()

# 3. Final Safety: Check if df is empty before training
if df.empty:
    st.error("Data fetch error: Nifty ka data abhi load nahi ho raha hai. Try again later.")
else:
    split_idx = int(len(df) * 0.50)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    features = ['Price', 'Kalman', 'Weighted_Momentum', 'Step_Momentum', 'Past_150_Diff']
    model = RandomForestRegressor(n_estimators=100, max_depth=5).fit(train[features], train['Target'])
    test['Predicted_150_Candle_Price'] = model.predict(test[features])

    st.subheader("📋 Discovery Engine (Live to July 2026)")
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
