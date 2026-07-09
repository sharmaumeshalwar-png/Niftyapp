import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import numpy as np

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Recursive Learning Engine")

@st.cache_data(ttl=3600)
def get_data():
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    return df.dropna()

df = get_data()

# Logic: Recursive Learning
# Hum har point (i) par predict karenge ki (i+150) kya hoga,
# aur compare karenge (i-150) ki purani prediction se.
results = []
# Model training sirf 150+ candle hone ke baad shuru hogi
for i in range(150, len(df) - 150):
    # Training set: Sirf wo data jo current 'i' se pehle ka hai
    train = df.iloc[:i]
    
    # Feature: Current Momentum
    X_train = train[['Price']].values
    y_train = df['Price'].iloc[150:i+150].values 
    
    # Model train at every step
    model = RandomForestRegressor(n_estimators=10).fit(X_train[:len(y_train)], y_train)
    
    # Prediction for 150 candles ahead
    current_val = df[['Price']].iloc[[i]]
    pred = model.predict(current_val)[0]
    
    # Reality (Actual price 150 candles later)
    actual = df['Price'].iloc[i+150]
    
    results.append({
        'Date': df.index[i],
        'Price_Then': df['Price'].iloc[i],
        'Prediction_For_150_Ahead': pred,
        'Actual_Result_150_Ahead': actual
    })

res_df = pd.DataFrame(results)

st.subheader("📋 Recursive Prediction & Learning Audit")
st.write("Model har candle par khud ko update kar raha hai.")

st.data_editor(
    res_df.sort_values(by='Date', ascending=False),
    use_container_width=True,
    column_config={
        "Price_Then": st.column_config.NumberColumn("Price at that time", format="%.2f"),
        "Prediction_For_150_Ahead": st.column_config.NumberColumn("ML Prediction", format="%.2f"),
        "Actual_Result_150_Ahead": st.column_config.NumberColumn("Actual Market", format="%.2f"),
    }
)
