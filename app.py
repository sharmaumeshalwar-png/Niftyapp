import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Page Setup
st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Discovery Engine [Real-Time Fixed]")

# =====================================================================
# MATH & DATA ENGINE
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
    ticker = "^NSEI"
    raw = yf.download(ticker, period="2y", interval="1h", progress=False)
    
    if raw.empty: return pd.DataFrame()
        
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # Mathematical Features
    df['Kalman'] = apply_kalman_filter_custom(df['Price'].values)
    df['Weighted_Momentum'] = apply_kalman_filter_custom((df['Price'] - df['Kalman']).values)
    df['Step_Momentum'] = np.round(apply_kalman_filter_custom(df['Weighted_Momentum'].values) * 10)
    df['Past_150_Diff'] = df['Price'] - df['Price'].shift(150)
    
    # FIXED: Future data target shift ki jagah, hum 'Price' ko hi target karenge
    # Target: Agli candle ka price
    df['Target'] = df['Price'].shift(-1) 
    return df.dropna()

df = get_ml_data()

# =====================================================================
# DASHBOARD & ML
# =====================================================================
if df.empty:
    st.error("Data load error.")
else:
    # 50:50 Split
    split_idx = int(len(df) * 0.50)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    
    features = ['Price', 'Kalman', 'Weighted_Momentum', 'Step_Momentum', 'Past_150_Diff']
    model = RandomForestRegressor(n_estimators=100, max_depth=5).fit(train[features], train['Target'])
    
    # Sirf present data par predict karega
    test['Predicted_Next_Price'] = model.predict(test[features])
    test['Projection_Date'] = test.index + pd.offsets.BusinessDay(1)

    st.subheader("📋 Discovery Engine (Real-Time Current Data)")
    st.data_editor(
        test.sort_index(ascending=False),
        use_container_width=True,
        column_config={
            "Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
            "Predicted_Next_Price": st.column_config.NumberColumn("ML Next Price", format="%.2f"),
            "Projection_Date": st.column_config.DateColumn("Projection Date", format="DD/MM/YYYY"),
        }
    )
