import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Discovery Pro", layout="wide")
st.title("🚀 Nifty 50 Discovery Momentum Engine")

# =====================================================================
# MATH ENGINES
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=100.0):
    x, p = data_array[0], initial_p
    q, r = 0.0001, 2.5
    res = []
    for z in data_array:
        p += q
        k = p / (p + r)
        x += k * (z - x)
        p = (1 - k) * p
        res.append(x)
    return res

# =====================================================================
# DATA ENGINE
# =====================================================================
@st.cache_data(ttl=3600)
def get_processed_data():
    raw = yf.download("^NSEI", period="2y", interval="1h")
    df = pd.DataFrame(index=raw.index)
    df[['Close', 'High', 'Low']] = raw[['Close', 'High', 'Low']].ffill()
    
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Gap'] = df['a_Close'] - df['b_Kalman']
    df['Normalized_Gap'] = df['c_Gap'] / (df['ATR'] + 1e-10)
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    return df.dropna()

df = get_processed_data()
split_idx = int(len(df) * 0.50)
predict = df.iloc[split_idx:].copy()

# Features & Model
features = ['c_Gap', 'Normalized_Gap', 'ATR']
model = RandomForestClassifier(n_estimators=150, max_depth=3).fit(df.iloc[:split_idx][features], df.iloc[:split_idx]['Target'])
predict['Prob_Up'] = model.predict_proba(predict[features])[:, 1]
predict['Discovery_Score'] = (predict['Prob_Up'] * predict['c_Gap']) / (predict['ATR'] + 1e-10)
predict['Scaled_Discovery_Momentum'] = apply_kalman_filter_custom(predict['Discovery_Score'].values, initial_p=0.50) * 100

# =====================================================================
# DASHBOARD WITH DUAL CHART
# =====================================================================
st.subheader("📊 Price vs Scaled Discovery Momentum")

# Streamlit line_chart automatically handle multiple columns
# Agar values bahut alag hain (Price 20000 vs Momentum 50), 
# toh aap "use_container_width" se zoom kar sakte hain.
st.line_chart(predict[['a_Close', 'Scaled_Discovery_Momentum']])

st.subheader("📋 Interactive Discovery Table")
display_df = predict[['a_Close', 'Prob_Up', 'ATR', 'Scaled_Discovery_Momentum', 'Discovery_Score']].sort_index(ascending=False)

st.data_editor(
    display_df,
    use_container_width=True,
    column_config={
        "a_Close": st.column_config.NumberColumn("Close Price", format="%.2f"),
        "Scaled_Discovery_Momentum": st.column_config.NumberColumn("Scaled Disc. Momentum", format="%.2f"),
        "Discovery_Score": st.column_config.NumberColumn("Discovery_Score", format="%.4f"),
    },
    height=500
)
