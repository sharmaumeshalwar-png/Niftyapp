import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Discovery Pro", layout="wide")
st.title("🚀 Nifty 50 Hybrid Discovery Engine [Gap-Aware]")

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
# DATA & HYBRID ENGINE
# =====================================================================
@st.cache_data(ttl=3600)
def get_data():
    raw = yf.download("^NSEI", period="2y", interval="1h")
    df = pd.DataFrame(index=raw.index)
    df[['Close', 'High', 'Low', 'Open']] = raw[['Close', 'High', 'Low', 'Open']].ffill()
    
    # Quantitative Features
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Gap'] = df['a_Close'] - df['b_Kalman']
    
    # Gap Analysis
    df['Prev_High_Low'] = (df['High'].shift(1) - df['Low'].shift(1))
    df['Gap_Value'] = df['Open'] - df['Close'].shift(1)
    df['Gap_Intensity'] = df['Gap_Value'] / (df['ATR'] + 1e-10)
    
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    return df.dropna()

df = get_data()
split_idx = int(len(df) * 0.50)
train, predict = df.iloc[:split_idx], df.iloc[split_idx:].copy()

# Machine Learning
features = ['c_Gap', 'ATR', 'Prev_High_Low', 'Gap_Value', 'Gap_Intensity']
model = RandomForestClassifier(n_estimators=150, max_depth=3).fit(train[features], train['Target'])
predict['Prob_Up'] = model.predict_proba(predict[features])[:, 1]

# Momentum calculation
predict['Discovery_Momentum'] = apply_kalman_filter_custom(predict['Prob_Up'].values, initial_p=0.50)
predict['Scaled_Momentum'] = predict['Discovery_Momentum'] * 100

# =====================================================================
# DASHBOARD
# =====================================================================
st.sidebar.header("📢 Market Events (Metadata)")
st.sidebar.write("• 10:00 AM: RBI Policy Decision")
st.sidebar.write("• 01:30 PM: Reliance Q1 Results")
st.sidebar.write("• *Note: Metadata is for reference only; not used in ML.*")

st.line_chart(predict[['a_Close', 'Scaled_Momentum']])

st.data_editor(
    predict[['a_Close', 'Prob_Up', 'Gap_Intensity', 'Scaled_Momentum']].sort_index(ascending=False),
    use_container_width=True,
    column_config={
        "a_Close": st.column_config.NumberColumn("Price", format="%.2f"),
        "Scaled_Momentum": st.column_config.NumberColumn("Momentum (Scaled)", format="%.2f"),
    },
    height=500
)
