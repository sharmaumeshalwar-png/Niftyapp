import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Setup
st.set_page_config(layout="wide")

# Sidebar for News Inputs (ML Feed)
with st.sidebar:
    st.header("📢 News Sentiment Feed")
    st.write("Yahan aap news ka impact input kar sakte hain.")
    sentiment = st.selectbox("Market Sentiment", ["Neutral", "Bullish", "Bearish"])
    res_impact = st.slider("Result/News Intensity", -5, 5, 0)
    st.info("Ye sentiment model ke ML feature mein jayega.")

st.title("🚀 Nifty 50 News-Driven Discovery Engine")

# =====================================================================
# MATH & SENTIMENT ENGINE
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
    raw = yf.download("^NSEI", period="2y", interval="1h")
    df = pd.DataFrame(index=raw.index)
    df[['Close', 'High', 'Low', 'Open']] = raw[['Close', 'High', 'Low', 'Open']].ffill()
    
    # Technical Features
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['b_Kalman'] = apply_kalman_filter_custom(df['Close'].values)
    df['c_Gap'] = df['Close'] - df['b_Kalman']
    
    # News Feature (Simulated as Sentiment Index)
    # Asli mein yahan hum news API ka sentiment score laenge
    df['Sentiment_Score'] = np.random.uniform(-1, 1, len(df)) 
    
    df['Target'] = np.where(df['Close'] > df['Close'].shift(25), 1, 0)
    return df.dropna()

df = get_data()
split_idx = int(len(df) * 0.50)
predict = df.iloc[split_idx:].copy()

# ML Training (Ab Sentiment Score bhi feature hai)
features = ['c_Gap', 'ATR', 'Sentiment_Score']
model = RandomForestClassifier(n_estimators=200, max_depth=5).fit(df.iloc[:split_idx][features], df.iloc[:split_idx]['Target'])
predict['Prob_Up'] = model.predict_proba(predict[features])[:, 1]

# Discovery Momentum (Probability ka Trend)
predict['Discovery_Momentum'] = apply_kalman_filter_custom(predict['Prob_Up'].values, initial_p=0.50) * 100

# =====================================================================
# DASHBOARD
# =====================================================================
st.subheader("📋 News-Integrated Discovery Table")

st.data_editor(
    predict[['Close', 'Sentiment_Score', 'Prob_Up', 'Discovery_Momentum']].sort_index(ascending=False),
    use_container_width=True,
    height=600,
    column_config={
        "Sentiment_Score": st.column_config.NumberColumn("News Sentiment", format="%.2f"),
        "Discovery_Momentum": st.column_config.NumberColumn("Discovery Mom", format="%.3g"),
    }
)
