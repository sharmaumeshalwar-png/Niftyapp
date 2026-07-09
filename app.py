import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Setup
st.set_page_config(page_title="Nifty Discovery Pro", layout="wide")

# Sidebar - Events (Calculation se bahar)
with st.sidebar:
    st.header("📢 Market Events")
    st.write("• 10:00 AM: RBI Policy Decision")
    st.write("• 01:30 PM: Reliance Q1 Result")
    st.write("• 03:00 PM: Global Market Cues")
    st.info("💡 Tip: Table ke columns ko drag karke move karein.")

st.title("🚀 Nifty 50 Discovery Pro Engine")

# =====================================================================
# 1. MATH & DATA ENGINE (The Full Logic)
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
def get_full_data():
    # 2 Year Data Download
    raw = yf.download("^NSEI", period="2y", interval="1h")
    df = pd.DataFrame(index=raw.index)
    df[['Close', 'High', 'Low', 'Open']] = raw[['Close', 'High', 'Low', 'Open']].ffill()
    
    # Discovery Metrics
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Gap'] = df['a_Close'] - df['b_Kalman']
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    return df.dropna()

# =====================================================================
# 2. ML & MOMENTUM CALCULATION
# =====================================================================
df = get_full_data()
split_idx = int(len(df) * 0.50)
train, predict = df.iloc[:split_idx], df.iloc[split_idx:].copy()

# Random Forest Training
features = ['c_Gap', 'ATR']
model = RandomForestClassifier(n_estimators=150, max_depth=3).fit(train[features], train['Target'])
predict['Prob_Up'] = model.predict_proba(predict[features])[:, 1]

# Calculations
predict['Weighted_Momentum'] = apply_kalman_filter_custom(predict['c_Gap'].values, initial_p=0.50)
predict['Step_Momentum'] = np.round(apply_kalman_filter_custom(predict['Weighted_Momentum'].values, initial_p=0.50) * 10)
predict['Discovery_Momentum'] = apply_kalman_filter_custom(predict['Prob_Up'].values, initial_p=0.50) * 100

# =====================================================================
# 3. INTERACTIVE DASHBOARD (Date & Moveable Columns)
# =====================================================================
st.subheader("📋 Discovery Table (Full Data)")

# Reset index taaki Date column ban jaye aur aap use move kar sako
display_df = predict[['a_Close', 'Prob_Up', 'Weighted_Momentum', 'Step_Momentum', 'Discovery_Momentum']].reset_index()

st.data_editor(
    display_df.sort_values(by='Datetime', ascending=False),
    use_container_width=True,
    height=600,
    column_config={
        "Datetime": st.column_config.DatetimeColumn("Date & Time", format="DD/MM/YYYY HH:mm"),
        "a_Close": st.column_config.NumberColumn("Price", format="%.2f"),
        "Prob_Up": st.column_config.ProgressColumn("Prob Up", format="%.2f", min_value=0, max_value=1),
        "Weighted_Momentum": st.column_config.NumberColumn("Weighted Mom", format="%.4f"),
        "Step_Momentum": st.column_config.NumberColumn("Step Mom", format="%.0f"),
        "Discovery_Momentum": st.column_config.NumberColumn("Discovery Mom", format="%.2f")
    }
)
