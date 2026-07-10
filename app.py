import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# Page Config
st.set_page_config(page_title="Nifty Engine", layout="wide")

# =====================================================================
# FAIL-SAFE DATA LOADER
# =====================================================================
def get_nifty_data():
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        # Ticker fix: Some regions require 'NSEI' instead of '^NSEI'
        data = yf.download("NSEI", start=start_date, end=end_date, interval="1h", progress=False)
        
        if data.empty:
            # Fallback to ^NSEI if NSEI fails
            data = yf.download("^NSEI", start=start_date, end=end_date, interval="1h", progress=False)
            
        return data
    except Exception as e:
        st.error(f"Critical Data Error: {e}")
        return None

# App Logic
st.title("📊 Nifty 50 Engine [Robust Mode]")
raw_df = get_nifty_data()

if raw_df is not None and not raw_df.empty:
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    
    df = raw_df[['Open', 'High', 'Low', 'Close']].ffill().dropna()
    st.success(f"✅ Data Loaded: {len(df)} candles found.")
    
    # [BAAKI SABHI KALMAN AUR ML LOGIC YAHAN RAKHEIN]
    # ...
else:
    st.warning("⚠️ Data download failed. Please check your internet connection or try again in a few minutes.")
