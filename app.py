import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50: Liquidity Hunt Tree Decoder")

@st.cache_data(ttl=3600)
def get_hunt_data():
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False).ffill()
    
    # 1. Geometry: Wick vs Body (Is it a sweep?)
    df['Wick_Ratio'] = (df['High'] - df['Low']) / (abs(df['Close'] - df['Open']) + 0.001)
    
    # 2. Hunting Logic: Price broke previous 5-hour low/high
    df['Prev_Low'] = df['Low'].shift(1).rolling(5).min()
    df['Prev_High'] = df['High'].shift(1).rolling(5).max()
    df['Is_SL_Hunt'] = ((df['Low'] < df['Prev_Low']) | (df['High'] > df['Prev_High'])).astype(int)
    
    # 3. Success: Kya hunt ke baad price reversal hua?
    df['Success'] = ((df['Close'].shift(-2) - df['Close']) * np.sign(df['Close'] - df['Open']) < 0).astype(int)
    
    return df.dropna()

data = get_hunt_data()
features = ['Wick_Ratio', 'Is_SL_Hunt']

split = int(len(data) * 0.5)
train, test = data.iloc[:split], data.iloc[split:]

# Tree Model: Jo har angle ko decode karega
model = DecisionTreeClassifier(max_depth=4)
model.fit(train[features], train['Success'])

st.subheader("📋 Tree Logic Audit")
st.write("Decision Tree har 'Hunt' ko analyze kar rahi hai ki wo kitni valid hai.")
st.dataframe(test.sort_index(ascending=False).head(15), use_container_width=True)

# Visualizing the Decision Logic
if st.button("Show Tree Logic Structure"):
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(model, feature_names=features, filled=True, ax=ax)
    st.pyplot(fig)
