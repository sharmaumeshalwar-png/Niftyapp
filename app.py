import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Recursive Discovery Engine [Fast]")

@st.cache_data(ttl=3600)
def get_recursive_data():
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # 150 Candle feature (Pichla 150 ghante ka trend)
    df['SMA_150'] = df['Price'].rolling(150).mean()
    
    # Target (150 candles aage)
    df['Actual_Future'] = df['Price'].shift(-150)
    return df.dropna()

df = get_recursive_data()

# Model training (Ek hi baar train hoga)
# Hum 70% data ko "Learning Memory" bana rahe hain
train_size = int(len(df) * 0.70)
train = df.iloc[:train_size]
model = RandomForestRegressor(n_estimators=50).fit(train[['Price', 'SMA_150']], train['Actual_Future'])

# Recursive Projection
# Har candle par model check kar raha hai ki 150 candles baad kya hona chahiye
df['Recursive_Prediction'] = model.predict(df[['Price', 'SMA_150']])
df['Target_Date'] = df.index + pd.offsets.BusinessDay(23)

st.subheader("📋 Recursive Prediction Audit")
st.write("Engine har ghante ke liye 150-candle future predict kar raha hai.")

st.data_editor(
    df.sort_index(ascending=False).head(100),
    use_container_width=True,
    column_config={
        "Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
        "Recursive_Prediction": st.column_config.NumberColumn("ML Prediction (150 Ahead)", format="%.2f"),
        "Actual_Future": st.column_config.NumberColumn("Reality (Actual)", format="%.2f"),
    }
)
