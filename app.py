import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

st.set_page_config(layout="wide")
st.title("🎯 2-Year Convergence Sniper [Hardened Version]")

@st.cache_data(ttl=3600)
def get_hardened_data():
    # Aaj ki date se 730 din pehle tak ka data
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=730)
    
    # Data load with explicit index handling
    df = yf.download("^NSEI", start=start, end=end, interval="1h", progress=False)
    
    # MultiIndex issue solve karne ke liye
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Index check and cleaning
    df = df.dropna(subset=['Close'])
    return df

df = get_hardened_data()

if df.empty:
    st.error("Data load nahi ho pa raha hai. Server connection limit hit ho sakti hai.")
else:
    st.write(f"📊 Total Records Found: {len(df)}")
    
    # Convergence Logic
    results = []
    # df.index is the Datetime index
    for i in range(8, len(df)):
        window = df.iloc[i-8:i]
        conv_price = window['Close'].mean()
        
        results.append({
            'Target_Time': df.index[i],
            'Convergence_Price': round(conv_price, 2),
            'Status': 'LOCK'
        })
    
    res_df = pd.DataFrame(results)
    st.dataframe(res_df.sort_values('Target_Time', ascending=False).head(50), use_container_width=True)
