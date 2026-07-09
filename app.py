import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Setup
st.set_page_config(layout="wide")

# Side-Panel
with st.sidebar:
    st.header("📢 Market Events")
    st.write("• 10:00 AM: RBI Policy Decision")
    st.write("• 01:30 PM: Reliance Q1 Result")
    st.write("• 03:00 PM: Global Market Cues")
    st.write("---")
    st.info("💡 Tip: 10 Saal ka Daily Data Load ho raha hai.")

st.title("🚀 Nifty 50 Discovery Pro [10Y Daily Engine]")

# =====================================================================
# MATH & DATA ENGINE
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
    return np.array(res)

@st.cache_data(ttl=3600)
def get_data():
    # 10 Saal ka Daily Data
    raw = yf.download("^NSEI", period="10y", interval="1d")
    df = pd.DataFrame(index=raw.index)
    df[['Close', 'High', 'Low', 'Open']] = raw[['Close', 'High', 'Low', 'Open']].ffill()
    
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Gap'] = df['a_Close'] - df['b_Kalman']
    # 10 saal ke liye 25 din ka target shift rakha hai
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    return df.dropna()

df = get_data()
# 50:50 Split for 10 years of data
split_idx = int(len(df) * 0.50)
predict = df.iloc[split_idx:].copy()

# ML Training
features = ['c_Gap', 'ATR']
model = RandomForestClassifier(n_estimators=150, max_depth=3).fit(df.iloc[:split_idx][features], df.iloc[:split_idx]['Target'])
predict['Prob_Up'] = model.predict_proba(predict[features])[:, 1]

# Calculations
predict['Prob_Weighted_Momentum'] = apply_kalman_filter_custom(predict['Prob_Up'].values, initial_p=0.50) * 100
predict['Prob_Step_Momentum'] = np.round(apply_kalman_filter_custom(predict['Prob_Weighted_Momentum'].values, initial_p=0.50) / 10)
predict['Kalman_Adjusted_Mom'] = predict['Prob_Weighted_Momentum'] * predict['b_Kalman']

# =====================================================================
# DASHBOARD
# =====================================================================
st.subheader("📋 10-Year Daily Discovery Table")
display_df = predict.reset_index()

st.data_editor(
    display_df.sort_values(by='Date', ascending=False), # Note: Daily data mein index 'Date' hota hai
    use_container_width=True,
    height=600,
    column_config={
        "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
        "Prob_Weighted_Momentum": st.column_config.NumberColumn("Prob Weighted Mom", format="%.3g"),
        "Prob_Step_Momentum": st.column_config.NumberColumn("Prob Step Mom", format="%.3g"),
        "Kalman_Adjusted_Mom": st.column_config.NumberColumn("Kalman Adjusted Mom", format="%.3g"),
    }
)
