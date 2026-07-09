import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

st.set_page_config(layout="wide")
st.title("🎯 2-Year Convergence Sniper [Live Audit]")

@st.cache_data(ttl=3600)
def get_clean_data():
    # 2 saal ka date range
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=730)
    
    # Data load
    df = yf.download("^NSEI", start=start, end=end, interval="1h", progress=False)
    
    # Data cleaning for blank issues
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index() # Index ko column mein badla
    df = df.ffill().dropna()
    return df

df = get_clean_data()

# Check agar data load hua
if len(df) == 0:
    st.error("Data load nahi hua, server connection check karo.")
else:
    st.write(f"📊 Total Records Found: {len(df)}")
    
    # 8-Candle Convergence Logic
    # Hum 8th index se start kar rahe hain taaki calculation error na ho
    results = []
    for i in range(8, len(df)):
        window = df.iloc[i-8:i]
        conv_price = window['Close'].mean()
        
        results.append({
            'Target_Time': df.iloc[i]['Datetime'] if 'Datetime' in df.columns else df.iloc[i]['index'],
            'Convergence_Price': round(conv_price, 2),
            'Status': 'LOCK'
        })
    
    res_df = pd.DataFrame(results)
    st.dataframe(res_df.sort_values('Target_Time', ascending=False).head(50), use_container_width=True)
