import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Discovery Pro", layout="wide")
st.title("🚀 Nifty 50 Advanced Discovery Engine [Pro Mode]")

# =====================================================================
# MATH ENGINES (Fixed & Optimized)
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

def apply_non_linear_kalman_momentum(data_array):
    x, p = data_array[0], 1.0
    q, r = 0.05, 0.2
    res = []
    for z in data_array:
        p += q
        k = p / (p + r)
        x += k * (z - x)
        p = (1 - k) * p
        res.append(x)
    return res

# =====================================================================
# DATA ENGINE WITH ATR INTEGRATION
# =====================================================================
@st.cache_data(ttl=3600)
def get_data():
    raw = yf.download("^NSEI", period="2y", interval="1h")
    df = pd.DataFrame(index=raw.index)
    df[['Close', 'High', 'Low']] = raw[['Close', 'High', 'Low']].ffill()
    
    # Discovery Metrics
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Gap'] = df['a_Close'] - df['b_Kalman']
    df['Normalized_Gap'] = df['c_Gap'] / (df['ATR'] + 1e-10)
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    return df.dropna()

df = get_data()
split_idx = int(len(df) * 0.50)
train, predict = df.iloc[:split_idx], df.iloc[split_idx:].copy()

# Training
features = ['c_Gap', 'Normalized_Gap', 'ATR']
model = RandomForestClassifier(n_estimators=150, max_depth=3).fit(train[features], train['Target'])
predict['Prob_Up'] = model.predict_proba(predict[features])[:, 1]

# Multi-Layer Columns
predict['Weighted_Momentum'] = apply_kalman_filter_custom(predict['c_Gap'].values, initial_p=0.50)
predict['Step_Momentum'] = np.round(apply_non_linear_kalman_momentum(predict['Weighted_Momentum'].values))
predict['Discovery_Score'] = (predict['Prob_Up'] * predict['Weighted_Momentum']) / (predict['ATR'] + 1e-10)

# =====================================================================
# INTERACTIVE DASHBOARD
# =====================================================================
st.subheader("📊 Discovery Engine: Interactive Live View")
st.write("💡 *Tip: Column headers ko click karke sort karein, ya drag karke move karein.*")

# Data Editor for Moveable Columns
st.data_editor(
    predict[['Prob_Up', 'ATR', 'Weighted_Momentum', 'Step_Momentum', 'Discovery_Score']].sort_index(ascending=False),
    use_container_width=True,
    column_config={
        "Prob_Up": st.column_config.ProgressColumn("Prob_Up", format="%.2f", min_value=0, max_value=1),
        "Discovery_Score": st.column_config.NumberColumn("Discovery_Score", format="%.4f"),
    },
    height=600
)

st.sidebar.metric("Latest Nifty Close", f"{df['a_Close'].iloc[-1]:.2f}")
st.sidebar.metric("Current ATR", f"{df['ATR'].iloc[-1]:.2f}")
