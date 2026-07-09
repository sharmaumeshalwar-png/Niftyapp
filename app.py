import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Discovery Pro", layout="wide")

# Sidebar for News (Metadata)
with st.sidebar:
    st.header("📢 Market Events (Metadata)")
    st.write("• 10:00 AM: RBI Policy Decision")
    st.write("• 01:30 PM: Reliance Q1 Results")
    st.write("• *Note: Metadata is for reference only; not used in ML.*")

st.title("🚀 Nifty 50 Hybrid Discovery Engine [Movable Views]")

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
    return res

@st.cache_data(ttl=3600)
def get_data():
    raw = yf.download("^NSEI", period="2y", interval="1h")
    df = pd.DataFrame(index=raw.index)
    df[['Close', 'High', 'Low', 'Open']] = raw[['Close', 'High', 'Low', 'Open']].ffill()
    
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Gap'] = df['a_Close'] - df['b_Kalman']
    df['Gap_Intensity'] = (df['Open'] - df['Close'].shift(1)) / (df['ATR'] + 1e-10)
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    return df.dropna()

df = get_data()
split_idx = int(len(df) * 0.50)
predict = df.iloc[split_idx:].copy()

features = ['c_Gap', 'ATR', 'Gap_Intensity']
model = RandomForestClassifier(n_estimators=150, max_depth=3).fit(df.iloc[:split_idx][features], df.iloc[:split_idx]['Target'])
predict['Prob_Up'] = model.predict_proba(predict[features])[:, 1]
predict['Scaled_Momentum'] = apply_kalman_filter_custom(predict['Prob_Up'].values, initial_p=0.50) * 100

# =====================================================================
# INTERACTIVE DASHBOARD
# =====================================================================
st.subheader("📊 Price vs Scaled Momentum")
st.line_chart(predict[['a_Close', 'Scaled_Momentum']])

st.subheader("📋 Interactive Discovery Table")
st.write("💡 *Date column (Index) hamesha pehla column hoga. Columns ko drag karke apni position set karein.*")

# Data Editor makes columns moveable by the user
st.data_editor(
    predict[['a_Close', 'Prob_Up', 'Gap_Intensity', 'Scaled_Momentum']].sort_index(ascending=False),
    use_container_width=True,
    column_config={
        "a_Close": st.column_config.NumberColumn("Close Price", format="%.2f"),
        "Scaled_Momentum": st.column_config.NumberColumn("Momentum (Scaled)", format="%.2f"),
    },
    height=500
)
