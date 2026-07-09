import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Setup
st.set_page_config(layout="wide")

st.title("🚀 Nifty 50 Classic Discovery Engine")

# =====================================================================
# CORE MATH ENGINE
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
def get_classic_data():
    raw = yf.download("^NSEI", period="2y", interval="1h")
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill()
    
    # Kalman Filter (0.50 logic applied via custom filter)
    df['Kalman'] = apply_kalman_filter_custom(df['Price'].values, initial_p=0.50)
    
    # Discovery Metrics
    df['Weighted_Momentum'] = apply_kalman_filter_custom((df['Price'] - df['Kalman']).values, initial_p=0.50)
    df['Step_Momentum'] = np.round(apply_kalman_filter_custom(df['Weighted_Momentum'].values, initial_p=0.50) * 10)
    
    # 25 Candle Past Comparison
    df['Past_Diff'] = df['Price'] - df['Price'].shift(25)
    
    return df.dropna()

df = get_classic_data()

# =====================================================================
# DASHBOARD
# =====================================================================
st.subheader("📋 Classic Discovery Table")

st.data_editor(
    df.sort_index(ascending=False),
    use_container_width=True,
    height=600,
    column_config={
        "Price": st.column_config.NumberColumn("Price", format="%.2f"),
        "Weighted_Momentum": st.column_config.NumberColumn("Weighted Mom", format="%.4f"),
        "Step_Momentum": st.column_config.NumberColumn("Step Mom", format="%.0f"),
        "Past_Diff": st.column_config.NumberColumn("25-Candle Diff", format="%.2f"),
    }
)
