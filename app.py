import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50: 2-Year Full-Range Consensus Audit (50:50)")

EMAS = [20, 50, 100, 200]

@st.cache_data(ttl=3600)
def get_full_consensus_data():
    # 2 saal ka pura data
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = df[['Close']].ffill()
    df.columns = ['Price']
    
    # Har EMA ke liye signal
    for e in EMAS:
        ema_val = df['Price'].ewm(span=e).mean()
        df[f'Signal_{e}'] = (ema_val.diff() > 0).astype(int)
        
    df['Consensus'] = df[[f'Signal_{e}' for e in EMAS]].sum(axis=1)
    df['Move'] = df['Price'].shift(-1) - df['Price']
    df['Target_Up'] = (df['Move'] > 0).astype(int)
    
    return df.dropna()

data = get_full_consensus_data()

# Strict 50:50 Split (1 saal train, 1 saal test)
split = int(len(data) * 0.5)
train = data.iloc[:split]
test = data.iloc[split:]

X = data[[f'Signal_{e}' for e in EMAS]]
y = data['Target_Up']

# Model Training (Sirf training data par)
model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
model.fit(X.iloc[:split], y.iloc[:split])

# Prediction (Audit: Pura 1 saal ka test data)
test = test.copy()
test['Prob_Up'] = model.predict(X.iloc[split:])
test['Action'] = test['Prob_Up'].apply(lambda x: "BUY" if x > 0.6 else ("SELL" if x < 0.4 else "HOLD"))

st.subheader(f"📋 2-Year Audit Result (Split at: {train.index[-1].date()})")
st.write(f"Training Range: {train.index[0].date()} to {train.index[-1].date()}")
st.write(f"Testing Range: {test.index[0].date()} to {test.index[-1].date()}")

st.dataframe(test.sort_index(ascending=False), use_container_width=True)
