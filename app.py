import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Discovery Pro", layout="wide")
st.title("🚀 Nifty 50 Discovery Momentum Engine")

# =====================================================================
# MATHEMATICAL ENGINES
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
# DATA & DISCOVERY ENGINE
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
train, predict = df.iloc[:split_idx], df.iloc[split_idx:].copy()

# Machine Learning
features = ['c_Gap', 'Normalized_Gap', 'ATR']
model = RandomForestClassifier(n_estimators=150, max_depth=3).fit(train[features], train['Target'])
predict['Prob_Up'] = model.predict_proba(predict[features])[:, 1]

# Discovery Score (Base)
predict['Discovery_Score'] = (predict['Prob_Up'] * predict['c_Gap']) / (predict['ATR'] + 1e-10)

# MOMENTUM ON DISCOVERY SCORE
# Discovery Score par Kalman lagaya
predict['Discovery_Momentum'] = apply_kalman_filter_custom(predict['Discovery_Score'].values, initial_p=0.50)
# Scaled for visibility (100x)
predict['Scaled_Discovery_Momentum'] = predict['Discovery_Momentum'] * 100

# =====================================================================
# DASHBOARD
# =====================================================================
st.subheader("📊 Discovery Score-Based Momentum")
st.line_chart(predict[['Scaled_Discovery_Momentum', 'Discovery_Score']])

display_df = predict[['a_Close', 'Prob_Up', 'Discovery_Score', 'Scaled_Discovery_Momentum']].sort_index(ascending=False)

st.data_editor(
    display_df,
    use_container_width=True,
    column_config={
        "a_Close": st.column_config.NumberColumn("Close Price", format="%.2f"),
        "Discovery_Score": st.column_config.NumberColumn("Discovery Score", format="%.4f"),
        "Scaled_Discovery_Momentum": st.column_config.NumberColumn("Scaled Disc. Momentum (100x)", format="%.2f"),
    },
    height=600
)
