import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Page Setup
st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Discovery ML Engine (150-Candle Trained)")

# =====================================================================
# MATH ENGINE
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
    raw = yf.download("^NSEI", period="2y", interval="1h")
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill()
    
    # Kalman & Momentum Logic
    df['Kalman'] = apply_kalman_filter_custom(df['Price'].values)
    df['Weighted_Momentum'] = apply_kalman_filter_custom((df['Price'] - df['Kalman']).values)
    df['Step_Momentum'] = np.round(apply_kalman_filter_custom(df['Weighted_Momentum'].values) * 10)
    
    # 150 Candle Past Aspect (Features for ML)
    df['Past_150_Diff'] = df['Price'] - df['Price'].shift(150)
    df['Target'] = df['Price'].shift(-150) # Agli 150 candle baad ka price target
    return df.dropna()

df = get_ml_data()

# ML Training
features = ['Price', 'Kalman', 'Weighted_Momentum', 'Step_Momentum', 'Past_150_Diff']
train_size = int(len(df) * 0.50)
model = RandomForestRegressor(n_estimators=100).fit(df.iloc[:train_size][features], df.iloc[:train_size]['Target'])

# Prediction
df['Predicted_150_Candle_Price'] = model.predict(df[features])

# =====================================================================
# DASHBOARD
# =====================================================================
st.subheader("📋 ML Predictive Discovery (150-Candle Outlook)")
st.write("ML model pichli 150 candles ka data seekh kar future ka price predict kar raha hai.")

st.data_editor(
    df.sort_index(ascending=False),
    use_container_width=True,
    height=500,
    column_config={
        "Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
        "Predicted_150_Candle_Price": st.column_config.NumberColumn("ML Pred. Target", format="%.2f"),
        "Past_150_Diff": st.column_config.NumberColumn("150-Candle Change", format="%.2f"),
    }
)
